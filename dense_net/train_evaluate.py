
from classifier import *
from dataset import *
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def train(EPOCH=10, data_loader=train_loader, is_pretrained=False, print_loss_gap=1):
    net = Classifier()
    net.to(cfg.device)
    writer = SummaryWriter("tensorboard")

    if is_pretrained:
        net.load_state_dict(torch.load(cfg.TEST_PTH + '\weights_epoch31.pth'))

    optimizer = optim.Adam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):

        print('{}th epoch starts'.format(str(epoch)))

        for idx, (window, y) in enumerate(data_loader):

            window, y = window.to(cfg.device), y.to(cfg.device)
            y_pred= net(window)
            loss = loss_fn(y_pred, y)

            if idx % print_loss_gap == 0:
                writer.add_scalars('loss/train', {'loss': loss}, global_step=epoch)

            if idx % 5 == 0:
                print_loss = loss.cpu().detach().item()
                print("epoch {} has finished {}% and the loss is {}".format(str(epoch), str(round(idx/1275, 4) * 100), str(print_loss)))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), cfg.TEST_PTH + '\weights_epoch{}.pth'.format(str(epoch + 32)))
    writer.flush()
    writer.close()

    return print("training process finished")

def cm_evaluate(dataset, weights_path):

    net_eval = Classifier()
    net_eval.to(cfg.device)
    net_eval.load_state_dict(torch.load(weights_path))
    net_eval.eval()
    pairs = []

    for idx, (window, y) in enumerate(dataset):
        label = y.item()
        window, y = window.to(cfg.device), y.to(cfg.device)
        y_pred = net_eval(window).to(cfg.device)
        y_pred = y_pred.squeeze(dim=0)
        y_pred = torch.argmax(y_pred).cpu().detach().item()
        pairs.append([label, y_pred])

    cm = np.zeros((cfg.CLAS, cfg.CLAS), dtype=int)
    for sample in pairs:
        cm[sample[0], sample[1]] += 1

    fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
    x_label_list, y_label_list = cfg.classes, cfg.classes
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(x_label_list)


    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(y_label_list)

    fig.suptitle('Confusion Matrix', fontsize=20)
    plt.xlabel('Target', fontsize=18)
    plt.ylabel('GT', fontsize=18)

    for i in range(cfg.CLAS):
        for j in range(cfg.CLAS):
            text = ax.text(j, i, cm[i, j], ha='center', va ='center', color='w')
            color_map = plt.imshow(cm)
            color_map.set_cmap('Blues')

    plt.savefig(r'C:\Users\liewei\Desktop\Confusion Matrix.png', dpi=300)
    return print('process finished')

if __name__ == '__main__':
    train(EPOCH=200, is_pretrained=True, data_loader=train_loader)
    # cm_evaluate(valid_loader, cfg.TEST_PTH + '\weights_epoch29.pth')
