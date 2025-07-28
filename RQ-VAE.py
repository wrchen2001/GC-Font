import sys
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import content_enc_builder
from model import dec_builder

from torch.utils.data import Dataset
from PIL import Image, ImageFile
from model.modules import weights_init
import pprint
import json
import tqdm

from torchvision.utils import make_grid, save_image


##############################################################################
# Phase 1: Model Definition - Components used in all phases
##############################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):

        super(VectorQuantizer, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

    def forward(self, inputs):
        # convert inputs from BCHW
        input_shape = inputs.shape
        # print(inputs.shape)

        # Flatten input ->[BC HW]
        flat_input = inputs.view(-1, self._embedding_dim)
        # print(flat_input.shape)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # print(distances.shape)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) 
        # print(encoding_indices.shape)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss  

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings
    

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, num_residual_quantizers = 3):
        super(ResidualVectorQuantizer, self).__init__()
        self._num_embeddings = num_embeddings 
        self._embedding_dim = embedding_dim    
        self._commitment_cost = commitment_cost   
        self._num_residual_quantizers = num_residual_quantizers   

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)  
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)  

        self._residual_embeddings = nn.ModuleList([
            nn.Embedding(self._num_embeddings, self._embedding_dim) for _ in range(num_residual_quantizers)
        ])  
        
        for embedding in self._residual_embeddings:
            embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)  
    
    def forward(self, inputs):
        inputs = inputs.to(device)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)  
        
        distances = (torch.sum(flat_input ** 2, dim = 1, keepdim=True) + 
                     torch.sum(self._embedding.weight ** 2, dim =1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim = 1).unsqueeze(1).to(device)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape).to(device)
    
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        all_encodings = encodings   

        for embedding in self._residual_embeddings:
            residual = inputs - quantized 
            flat_residual = residual.view(-1, self._embedding_dim)
            distances = (torch.sum(flat_residual ** 2, dim = 1, keepdim = True) + 
                         torch.sum(embedding.weight **2, dim =1) -
                         2 * torch.matmul(flat_residual, embedding.weight.t()))
            
            encoding_indices = torch.argmin(distances, dim =1).unsqueeze(1).to(device)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
            encodings.scatter_(1, encoding_indices, 1)

            quantized_residual = torch.matmul(encodings, embedding.weight).view(input_shape).to(device)  
            quantized = quantized + quantized_residual   

            all_encodings = all_encodings + encodings

            e_residual_latent_loss = F.mse_loss(quantized_residual.detach(), residual)
            q_residual_latent_loss = F.mse_loss(quantized_residual, residual.detach())

            loss = loss + q_residual_latent_loss + self._commitment_cost * e_residual_latent_loss
    

        quantized = inputs + (quantized - inputs).detach()   
        avg_probs = torch.mean(all_encodings/(self._num_residual_quantizers + 1), dim = 0)  
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings



class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__() 

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon  


    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)


        if self.training: 
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay = 0):
        super(Model, self).__init__()
        self._encoder = content_enc_builder(1,32,256)  

        if decay > 0.0:  
            self._rq_vae = ResidualVectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        else:
            self._rq_vae = ResidualVectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
            
        self._decoder = dec_builder(32, 1)  


    def forward(self, x):
        z = self._encoder(x) 
        loss, quantized, perplexity, _ = self._rq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity


class CombTrain_RQ_VAE_dataset(Dataset):
    def __init__(self, root, transform = None):
        self.img_path = root
        self.transform = transform
        self.imgs = self.read_file(self.img_path)
        # img = Image.open(self.imgs[0])
        # img = self.transform(img)
        # print(img.shape)


    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def __getitem__(self, index):
        img_name = self.imgs[index]
        #print(img_name[-5:-4])
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img) #Tensor [C H W] [1 128 128]
        return img

    
    def __len__(self):  
        return len(self.imgs)
    

##############################################################################
# Phase 2: Training the RQ-VAE Model
##############################################################################

num_training_updates = 50000
embedding_dim = 256
num_embeddings = 100
commitment_cost = 0.25
decay = 0
learning_rate = 2e-4

model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model.apply(weights_init("xavier"))
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

train_imgs_path = 'path/to/save/train_content_imgs/'
tensorize_transform = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

train_dataset = CombTrain_RQ_VAE_dataset(train_imgs_path, transform=tensorize_transform)

train_loader = DataLoader(train_dataset, batch_size=64, batch_sampler=None, drop_last=True, pin_memory=True, shuffle=True)

model.train()
train_res_recon_error = []
train_res_perplexity = []
train_rq_loss = []

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


for i in range(num_training_updates):
    data = next(iter(train_loader))
    train_data_variance = torch.var(data)
    data = data - 0.5 # normalize to [-0.5, 0.5]
    data = data.to(device)
    optimizer.zero_grad()

    rq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / train_data_variance
    loss = recon_error + rq_loss

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Track training metrics
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())
    train_rq_loss.append(rq_loss.item())

    if (i + 1) % 1000 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-1000:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-1000:]))
        print('rq_loss: %.3f' % np.mean(train_rq_loss[-1000:]))
        print()


##############################################################################
# Phase 3: Validation/Testing the Trained Model
##############################################################################


print("\nStarting model validation...")

# Validation dataset setup
val_imgs_path = 'path/to/save/val_content_imgs'
tensorize_transform = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])
val_dataset = CombTrain_RQ_VAE_dataset(val_imgs_path, transform=tensorize_transform)
validation_loader = DataLoader(val_dataset, batch_size=8, batch_sampler=None, drop_last=True, pin_memory=True, shuffle=True)


def val_(model,validation_loader):
    """Run validation and return original/reconstructed images"""
    model.eval()
    valid_originals = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)

    # Get model outputs
    rq_output_eval = model._encoder(valid_originals)
    _, valid_quantize, _, _ = model._rq_vae(rq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)
    return valid_originals, valid_reconstructions


# Run validation
org, recon_out = val_(model, validation_loader)
# show(make_grid((org+0.5).cpu().data), )
# show(make_grid((recon_out+0.5).cpu().data), )

# Save trained model
print("\nSaving trained model...")
torch.save(model,'/pretrained_weights/RQ-VAE_chn_.pth')    
torch.save(model.state_dict(),'./pretrained_weights/RQ-VAE_Parms_chn_.pth')


##############################################################################
# Phase 4: Calculating Character Similarities Using Trained Encoder
##############################################################################


print("\nCalculating character similarities using trained encoder...")

# Load model for feature extraction
embedding_dim, num_embeddings, commitment_cost, decay = 256, 100, 0.25, 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
models = torch.load('/pretrained_weights/RQ-VAE_chn_.pth')
encoder = models._encoder.to(device)
encoder.requires_gradq = False
encoder.to("cpu")


class CombTrain_RQ_VAE_dataset(Dataset):
    def __init__(self, root, transform = None):
        self.img_path = root
        self.transform = transform
        self.imgs = self.read_file(self.img_path)
        # img = Image.open(self.imgs[0])
        # img = self.transform(img)
        # print(img.shape)


    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list


    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img) #Tensor [C H W] [1 128 128]
        ret =(img_name, 
              img
        )
        return ret

    def __len__(self):

        return len(self.imgs)


# Setup similarity calculation
train_imgs_path = 'path/to/save/all_content_imgs'
tensorize_transform = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

batch = 3500 # all content imgs

sim_dataset = CombTrain_RQ_VAE_dataset(train_imgs_path, transform=tensorize_transform)
sim_loader = DataLoader(sim_dataset, batch_size=batch, batch_sampler=None, drop_last=False, pin_memory=True)  
similarity = []


def CosineSimilarity(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


while True:
    data = next(iter(sim_loader))  
    img_name = data[0]
    # print(data[0])
    img_tensor = data[1]
    img_tensor = img_tensor - 0.5 # normalize to [-0.5, 0.5]
    img_tensor = img_tensor.to("cpu")
    
    content_feature = encoder(img_tensor)
    vector = content_feature.view(content_feature.shape[0], -1)
    
    sim_all = {}
    for i in range(0, batch):  
        char_i = hex(ord(img_name[i][-5]))[2:].upper()
        dict_sim_i = {char_i:{}}
        for j in range(0,batch):
            char_j = hex(ord(img_name[j][-5]))[2:].upper()
            similarity = CosineSimilarity(vector[i],vector[j])
            if i==j:
                similarity=1.0
            sim_i2j = {char_j:float(similarity)}
            dict_sim_i[char_i].update(sim_i2j)
        sim_all.update(dict_sim_i)

    # Save similarity matrix to JSON    
    dict_json=json.dumps(sim_all) 

    with open('/pretrained_weights/all_char_similarity_unicode.json','w+') as file:
        file.write(dict_json)    
    break



