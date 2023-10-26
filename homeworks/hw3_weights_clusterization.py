import torch
import torch.nn as nn
from src.upd_scrfd.scrfd import SCRFD_500M
import src.clustering as clustering
import src.prunenet as prunenet


def gen(target, conv1x1=False):
    """a generator for iterating default Conv2d"""

    def _criterion(m):
        if isinstance(m, nn.Conv2d):
            if conv1x1:
                return m.kernel_size[0] * m.kernel_size[1] == 1
            else:
                return m.kernel_size[0] * m.kernel_size[1] > 1
        elif isinstance(m, nn.ConvTranspose2d):
            return True

        return False

    gen = (m for m in target.modules() if _criterion(m))

    return gen


def split_set(gen, n_convs, multi):
    """
        multi clustering examples

        --multi full:
            use a set of centroids for all kernels in the network

        --multi split:
            use a set of centroids for each layer in the network

        --multi 0-0=64,1-16=256:
            use 64 centroids for conv0 ~ conv0
            use 256 centroids for conv1 ~ conv16

            you can also try different number of centroids, different splits
    """
    if multi.find('full') >= 0:
        split = {
            'k_kernels': [int(multi.split('-')[-1])],
            'map': {i: 0 for i in range(n_convs)}
        }
    elif multi == 'split':
        kernels = [m.in_channels * m.out_channels for m in gen]
        split = {
            'k_kernels': [min(k, k_kernels) for k in kernels],
            'map': {i: i for i in range(n_convs)}
        }
    else:
        split = {'k_kernels': [], 'map': {}}
        for mi, m in enumerate(multi.split(',')):
            entry, k_kernels_split = m.split('=')
            split['k_kernels'].append(int(k_kernels_split))
            if entry.find('+') >= 0:
                entry = [int(e) for e in entry.split('+')]
            elif entry.find('-') >= 0:
                lb, ub = [int(e) for e in entry.split('-')]
                entry = [i for i in range(lb, ub + 1)]

            for e in entry:
                split['map'][e] = mi

    return split


def extract(split, parent, scale_type):
    k_list = [[] for _ in split['k_kernels']]
    s_dict = {}
    modules = [m for m in gen(parent)]

    for k, v in split['map'].items():
        kernels, scales = preprocess_kernels(
            modules[k],
            scale_type
        )
        k_list[v].extend(list(kernels))
        s_dict[k] = nn.Parameter(scales, requires_grad=(scale_type.find('train') >= 0))

    return k_list, s_dict


def preprocess_kernels(m, scale_type):
    '''
        return
            kernels:
                normalized kernels

            magnitudes:
                scales
    '''
    c_out, c_in, kh, kw = m.weight.size()
    weights = m.weight.data.view(c_out, c_in, kh * kw)
    cuts = m.cut.view(c_out, c_in, 1).byte()

    if scale_type.find('kernel') >= 0:
        magnitudes = weights.norm(2, dim=2)  # c_out x c_in

        if scale_type.find('norm') >= 0:
            magnitudes.mul_(weights[:, :, (kh * kw) // 2].sign())
        else:
            magnitudes.fill_(1)

        magnitudes.unsqueeze_(-1)  # c_out x c_in x 1
        kernels = weights.div(magnitudes)  # c_out x c_in x (kh * kw)
        magnitudes.mul_(cuts.float()).unsqueeze_(-1)  # c_out x c_in x 1 x 1

    kernels = kernels.masked_select(cuts).view(-1, kh, kw)

    return kernels, magnitudes


def get_centroids(
        split, k_list, n_init, max_iter, symmetry='i', n_GPUs=1):
    '''
        k-means clustering using custom module
        see clustering.py for details
    '''

    idx_list = []
    for i, kernels in enumerate(k_list):
        entry = [k for k, v in split['map'].items() if v == i]
        print('\nCluster {}'.format(i))
        print('Conv {}'.format(', '.join(str(e) for e in entry)))

        k_kernels = split['k_kernels'][i]
        kernel_stack = torch.stack(kernels)
        kh, kw = kernel_stack.size()[-2:]

        cluster = clustering.Clustering(
            k_kernels,
            n_init=n_init,
            max_iter=max_iter,
            symmetry=symmetry,
            cpu=True,
            n_GPUs=n_GPUs
        )
        centroids, idx = cluster.fit(kernel_stack.view(-1, kh * kw))
        centroids = centroids.view(-1, kh, kw)
        idx_tic = idx.div(k_kernels)
        centroids, idx = clustering.sort_clustering(
            centroids, idx % k_kernels, k_kernels
        )
        idx_list.append(idx + k_kernels * idx_tic)
        # self.register_parameter(
        #     'centroids_{}'.format(i), nn.Parameter(centroids)
        # )
        save_clustering_results(i, kernel_stack, centroids, idx, save='clustering')

    return idx_list


# def quantize_kernels(split, parent, idx_list, s_dict, debug=False):
#     modules_parent = [m for m in self.gen(parent)]
#     modules_self = [m for m in cconv.gen_cconv(self)]
#
#     idx_acc = [0] * len(split['k_kernels'])
#     for k, v in split['map'].items():
#         source = modules_parent[k]
#         target = modules_self[k]
#         c_out, c_in, _, _ = source.weight.size()
#         idx = torch.LongTensor(c_out * c_in)
#         for i, is_cut in enumerate(source.cut.view(-1)):
#             if is_cut != 0:
#                 idx[i] = idx_list[v][idx_acc[v]]
#                 idx_acc[v] += 1
#             # ignore pruned kernels
#             else:
#                 idx[i] = 0
#
#         target.set_params(
#             source,
#             centroids=getattr(self, 'centroids_{}'.format(v)),
#             idx=idx,
#             scales=s_dict[k],
#             debug=debug
#         )


def save_clustering_results(i, kernel_stack, centroids, idx, save, debug=False):
    _, kh, kw = kernel_stack.size()
    k_kernels = centroids.size(0)

    savedir = '../results/{}'.format(save)
    clustering.save_kernels(centroids, '{}/iter000_c{:0>2}.png'.format(savedir, i))
    clustering.save_distribution(idx, '{}/distribution_{:0>2}.pdf'.format(savedir, i))
    torch.save(idx, '{}/labels_{:0>2}.pt'.format(savedir, i))
    # visualize full kernels and their centroids
    if debug:
        for t in range(k_kernels):
            mask = (idx == t)
            mask_idx = torch.arange(len(kernel_stack))
            mask_idx = mask_idx.masked_select(mask).long()

            ref = centroids[t].view(1, kh, kw)
            collect = torch.cat((ref, kernel_stack[mask_idx]), dim=0)
            clustering.save_kernels(collect, '{}/vis_c{:0>2}_{:0>4}.png'.format(savedir, i, t), highlight=True)


if __name__ == '__main__':
    multi = 'full-256'
    symmetry = 'i'
    scale_type = 'kernel_norm_train'
    n_init = 1
    max_iter = 4500

    k = torch.load('../results/clustering/labels_00.pt')
    model = SCRFD_500M(nc=1)
    model.load_from_checkpoint('../weights/upd_SCRFD_500M_KPS.pth')

    prunenet.register_cut(model)

    n_convs = sum(1 for m in gen(model))
    n_params = sum(m.weight.nelement() for m in gen(model))
    print('{} 3x3 convs / {} params.'.format(n_convs, n_params))

    # split set for multi clustering
    split = split_set(gen(model), n_convs, multi)

    k_list, s_dict = extract(split, model, scale_type)

    # clustering
    idx_list = get_centroids(
        split,
        k_list,
        n_init,
        max_iter,
        symmetry=symmetry,
        n_GPUs=1
    )

    # # replace kernels with centroids
    # self.quantize_kernels(parent, idx_list, s_dict, debug=args.debug)
    #
    # model_name = "bert-base-uncased"
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # bert_model = TFBertModel.from_pretrained(model_name)
    # print('done')