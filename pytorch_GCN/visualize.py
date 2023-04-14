
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from random import randrange

"""Visualization"""
# visualize learned feature representation
def visualize_learnedFeature_tSNE(labels, out_features, dataset):
    color_map = {0: "red", 1: "blue", 2: "green",
                           3: "orange", 4: "yellow", 5: "pink", 6: "gray"}

    if dataset =='citeseer':
        num_classes = 6
    elif dataset == 'cora':
        num_classes = 7
    elif dataset =='pubmed': 
        num_classes = 3                    
    node_labels = labels.cpu().numpy()
    out_features = out_features.cpu().detach().numpy()
    t_sne_X = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(out_features)

    plt.figure()
    for class_id in range(num_classes):
        plt.scatter(t_sne_X[node_labels == class_id, 0],
                    t_sne_X[node_labels == class_id, 1], s=20,
                    color=color_map[class_id],
                    edgecolors='black', linewidths=0.15)

    plt.axis("off")
    plt.title("t-SNE projection of the learned features for "+dataset)
    plt.show()

# visulaize validation loss and accuracy 
def visualize_validation_performance(val_acc, val_loss):
    f, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(val_loss, linewidth=2, color="red")
    ax[0].set_title("Validation loss")
    ax[0].set_ylabel("Cross Entropy Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].grid()
    ax[1].plot(val_acc, linewidth=2, color="red")
    ax[1].set_title("Validation accuracy")
    ax[1].set_ylabel("Acc")
    ax[1].set_xlabel("Epoch")
    ax[1].grid()
    plt.show()



