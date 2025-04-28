from scipy import stats
import os
import torch
import numpy as np
import random
import sys
import sklearn.metrics.pairwise as pdist
import sklearn.metrics as skmetrics
import sklearn.cluster as skcluster
# import matplotlib.pyplot as plt
import scipy.optimize as optimize

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
random.seed(0)

# Choosing `num_centers` random data points as the initial centers
def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device_gpu)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(1.5e9 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device_gpu)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes

# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device_gpu)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device_gpu)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device_gpu))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device_gpu))
    centers /= cnt.view(-1, 1)
    return centers

def cluster(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            sys.stdout.write('\n')
            print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
    return centers, codes

def get_code2spk_spk2code(uttcmb, pudo_la):
    utt2psdlb = {utt:label for utt, label in zip(uttcmb, pudo_la)}
    psdlb2spk = {}
    for utt in utt2psdlb:
        spk = int(utt.split('-')[0][2:])
        psdlb = utt2psdlb[utt]
        if psdlb in psdlb2spk:
            psdlb2spk[psdlb].append(spk)
        else:
            psdlb2spk[psdlb] = [spk]
    
    spk2psdlb = {}
    for i in psdlb2spk:
        for j in psdlb2spk[i]:
            if j in spk2psdlb:
                spk2psdlb[j].append(i)
            else:
                spk2psdlb[j] = [i]
    return psdlb2spk, spk2psdlb

def nmi(pseudo_label, utt, data_name):
    l_map = {l:i for i, l in enumerate(set(pseudo_label))}
    lp = [l_map[l] for l in pseudo_label]

    utt2lt = {l.split()[0]:l.split()[0].split('-')[0] for l in open('data/%s/utt2spk' % data_name)}
    lt = [utt2lt[i] for i in utt]
    l_map = {l:i for i, l in enumerate(set(lt))}
    lt = [l_map[l] for l in lt]
    
    return skmetrics.normalized_mutual_info_score(lt, lp)

def get_embd_utt(system, data_name, typ):
    embd = np.load('exp/%s/%s_%s.npy' % (system, data_name, typ))
    utt = [line.split()[0] for line in open('data/%s/wav.scp' % data_name)]
    embd = embd / np.linalg.norm(embd, axis=1).reshape([-1, 1])
    return embd, np.array(utt)

def within_sum_of_squares(data, centroids, labels):
    SSW = 0
    for l in np.unique(labels):
        data_l = data[labels == l]
        resid = data_l - centroids[l]
        SSW += (resid**2).sum()
    return SSW

def between_cluster_variation(data, centroids, labels):
    B = 0
    c = data.mean(axis=0)
    for l in np.unique(labels):
        n = (labels == l).sum()
        resid = centroids[l] - c
        B = B + n * ((resid**2).sum())
    return B

def CH_idx(W, B, n, K):
    return (B / (K - 1)) / (W / (n - K))

def nmi_acc_pur(lp):
    lt = [l.split()[0].split('-')[0] for l in open('data/vox2dev/wav.scp')]
    l_map = {l:i for i, l in enumerate(set(lt))}
    lt = [l_map[l] for l in lt]

    l_map = {l:i for i, l in enumerate(set(lp))}
    lp = [l_map[l] for l in lp]

    nmi = skmetrics.normalized_mutual_info_score(lt, lp)

    num_k = max((len(set(lp)), len(set(lt))))
    cost = np.zeros((num_k, num_k))
    for i, k in zip(lp, lt):
        cost[i, k] = cost[i, k] + 1
    map_lab = optimize.linear_sum_assignment(-cost)[1]
    lp_map = np.array([map_lab[i] for i in lp])
    acc = np.sum(lp_map==lt) / len(lt)

    purity = []
    lp, lt = np.array(lp), np.array(lt)
    for i in set(lp):
        distri = lt[lp == i]
        purity.append(np.max([np.sum(distri == j) for j in set(distri)])/len(distri))
    purity = np.mean(purity)
    
    return '  NMI %.4f  Acc %2.2f  Pur %2.2f  ' % (nmi, acc*100, purity*100)


system = ''
data_name = 'vox2dev'
embd_dict = np.load('exp/%s/%s.npy' % (system, data_name), allow_pickle=True)
embd = []
for index, utt in enumerate(embd_dict.item()):
    embd.append(embd_dict.item().get(utt)[0])
embd = np.array(embd) 
embd = embd / np.sqrt(np.sum(embd * embd, axis=1))[:, None]

for num_centers in [8000]:
    device_gpu = torch.device('cuda')
    dataset = torch.from_numpy(embd.astype(np.float32)).to(device_gpu)
    centers, codes = cluster(dataset, num_centers)
    codes = codes.cpu().numpy()
    centers = centers.cpu().numpy()

np.save('exp/%s/%s_kmeans_%d_centers' % (system, data_name, num_centers), centers)
np.save('exp/%s/%s_kmeans_%d_codes' % (system, data_name, num_centers), codes)

utt = [line.split()[0] for line in open('data/vox2dev/wav.scp')]
f = open('data/dev_vox2_dino_ali_k8000/utt2spk', 'w')
for u, l in zip(utt, codes):
    f.write('%s %d\n' % (u, l))
    f.flush()
f.close()

lp = [l.strip().split()[1] for l in open('data/dev_vox2_dino_ali_k8000/utt2spk')]
ans = nmi_acc_pur(lp)
print(ans)