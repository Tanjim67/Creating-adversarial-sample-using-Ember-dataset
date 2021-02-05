import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from IPython.display import clear_output
import plots


class EmberNet(nn.Module):

    def __init__(self, device, scaler):
        super().__init__()
        self.conv_11 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64, stride=64, padding=0)
        self.norm_12 = nn.BatchNorm1d(128)
        self.drop_13 = nn.Dropout(0.1)
        self.conv_14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.norm_15 = nn.BatchNorm1d(128)
        self.drop_16 = nn.Dropout(0.1)

        self.linr_21 = nn.Linear(2304, 256)
        self.norm_22 = nn.BatchNorm1d(256)
        self.drop_23 = nn.Dropout(0.1)
        self.linr_24 = nn.Linear(256, 32)
        self.norm_25 = nn.BatchNorm1d(32)
        self.linr_out = nn.Linear(32, 2)
        self.hist = []
        self.device = device
        self.to(device)
        self.scaler = scaler

    def forward(self, x):
        x1 = F.relu(self.conv_11(x))
        x1 = self.norm_12(x1)
        x1 = self.drop_13(x1)
        x1 = F.relu(self.conv_14(x1))
        x1 = self.norm_15(x1)
        x1 = self.drop_16(x1)

        x2 = x1.reshape(-1, 2304)
        x2 = F.relu(self.linr_21(x2))
        x2 = self.norm_22(x2)
        x2 = self.drop_23(x2)
        x2 = F.relu(self.linr_24(x2))
        x2 = self.norm_25(x2)
        out_x = torch.sigmoid(self.linr_out(x2))

        return out_x

    def loss_batch(self, loss_func, xb, yb, opt=None, pred=None):
        if pred is None:
            pred = self(xb)
        loss = loss_func(pred, yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb)

    def eval_batch(self, xb, yb, loss_func, pred=None):
        if pred is None:
            pred = self(xb)
        corr = pred.argmax(dim=1).eq(yb).sum().item()
        loss = loss_func(pred, yb)
        return loss, corr, len(xb)

    def eval_batches(self, loss_func, dl):
        with torch.no_grad():
            losses, corr, nums = zip(
                *[self.eval_batch(xb.to(self.device), yb.to(self.device), loss_func) for xb, yb, _ in dl]
            )
        return np.sum(np.multiply(losses, nums)) / np.sum(nums), np.sum(corr) / np.sum(nums)

    def fit(self, epochs, loss_func, opt, train_dl, valid_dl, save_path, epsilon: float = None):
        old_epochs = len(self.hist)
        epochs = old_epochs + epochs
        for epoch in range(old_epochs, epochs):
            self.train()
            with tqdm.auto.tqdm(train_dl, unit="batch") as tepoch:
                for xb, yb, _ in tepoch:
                    r = np.random.randint(3, size=1).item()
                    tepoch.set_description(f"Epoch {epoch}")
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    if epsilon is None or r == 0:
                        self.loss_batch(loss_func, xb, yb, opt)
                    else:
                        xb_pert = self.attack_batch(xb, yb, loss_func, epsilon, constraints=True, mask_only=True)
                        self.loss_batch(loss_func, xb_pert, yb, opt)
                self.eval()
                train_loss, train_acc = self.eval_batches(loss_func, train_dl)
                val_loss, val_acc = self.eval_batches(loss_func, valid_dl)
                self.hist.append((epoch, train_loss, train_acc, val_loss, val_acc))
                tepoch.close()
            self.plot()
            print("Epoch {}:\t Train Loss= {}\t Train Accuracy= {}\t Validation Loss= {}\t Validation Accuracy= {}"
                  .format(epoch, train_loss, train_acc, val_loss, val_acc))
            if self.__multiple(epochs, 5):
                self.save(save_path)

    def __multiple(self, m, n):
        return True if m % n == 0 else False

    def test(self, loss_func, test_dl):
        self.eval()
        test_loss, test_acc = self.eval_batches(loss_func, test_dl)
        print("Test Loss= {}\t Test Accuracy= {}".format(test_loss, test_acc))
        return test_loss, test_acc

    def attack(self, loss_func, eps, adv_dl, constraints=False, malicous_only=False):
        adv_examples = []
        losses = []
        corrs = []
        nums = []
        self.eval()
        orig_loss, orig_acc = self.eval_batches(loss_func, adv_dl)
        for xb, yb, hashb in tqdm.auto.tqdm(adv_dl):
            xb, yb, hashb = xb.to(self.device), yb.to(self.device), hashb.to(self.device)
            if malicous_only:
                xb = xb[yb == 1]
                yb = yb[yb == 1]
            xb_pert = self.attack_batch(xb, yb, loss_func, eps, constraints)
            loss, corr, num = self.eval_batch(xb_pert, yb, loss_func)
            losses.append(loss)
            corrs.append(corr)
            nums.append(num)
            adv_examples.extend(list(
                zip(xb.cpu().detach(), xb_pert.cpu().detach(), hashb.cpu().detach(), yb.cpu().detach(),
                    self(xb).argmax(dim=1).cpu(), self(xb_pert).argmax(dim=1).cpu())))
        pert_loss, pert_acc = np.sum(np.multiply(losses, nums)) / np.sum(nums), np.sum(corrs) / np.sum(nums)
        print("Original Loss= {}\t Original Accuracy= {}\t Adverserial Loss= {}\t Adverserial Accuracy= {}"
              .format(orig_loss, orig_acc, pert_loss, pert_acc))
        return adv_examples, pert_acc

    def attack_batch(self, xb, yb, loss_func, eps, constraints, mask_only=False):
        xb.requires_grad = True
        loss = loss_func(self(xb), yb)
        self.zero_grad()
        loss.backward()
        grad_sign = xb.grad.sign()
        xb_pert = self.perturb(xb, eps, grad_sign, constraints, mask_only)
        self.zero_grad()
        return xb_pert

    def perturb(self, xb, eps, grad_sign, constraints=False, mask_only=False):
        if constraints is False:
            mask = torch.linspace(1, 1, steps=2381).to(self.device)
        else:
            mask = 256 * [0] + 256 * [0] + 104 * [1] + 10 * [1] + [1] + 50 * [0] + 11 * [1] + 255 * [0] + 1280 * [
                0] + 128 * [0] + 30 * [0]
            mask = torch.IntTensor(mask).to(self.device)

        xb_pert = xb + eps * grad_sign * mask

        if constraints and not mask_only:
            xb_pert = self.check_bounds(xb_pert, xb, mask)

        return xb_pert

    def check_bounds(self, xb_pert, xb, mask):
        scaler = self.scaler
        xb_pert_rescaled = scaler.inverse_transform(
            xb_pert.view(xb_pert.shape[0], xb_pert.shape[2]).cpu().detach().numpy())
        xb_rescaled = scaler.inverse_transform(xb.view(xb.shape[0], xb.shape[2]).cpu().detach().numpy())
        # check that everything is still in bounds
        del xb_pert
        only_bigger = [616, 617, 619, 620, 625, 685, 686, 687]
        only_booleans = [618, 621, 622, 623, 624]
        for i in range(len(xb_pert_rescaled)):
            for j in range(len(xb_pert_rescaled[i])):
                # not smaller than zero
                if xb_pert_rescaled[i][j] < 0 and mask[j]:
                    xb_pert_rescaled[i][j] = 0
                # not allowed to get smaller
                if xb_rescaled[i][j] > xb_pert_rescaled[i][j] and j in only_bigger:
                    xb_pert_rescaled[i][j] = xb_rescaled[i][j]
                elif xb_pert_rescaled[i][j] not in [0,1] and j in only_booleans:  # only boolean values allowed, cap at 0 and 1
                    if xb_rescaled[i][j] == 0:
                        if xb_pert_rescaled[i][j] > xb_rescaled[i][j]:
                            xb_pert_rescaled[i][j] = 1
                        else:
                            xb_pert_rescaled[i][j] = 0
                    else:
                        if xb_pert_rescaled[i][j] > xb_rescaled[i][j]:
                            xb_pert_rescaled[i][j] = 1
                        else:
                            xb_pert_rescaled[i][j] = 0

        return torch.from_numpy(np.expand_dims(scaler.transform(xb_pert_rescaled), axis=1)).to(self.device)

    def plot(self):
        clear_output(wait=True)
        plots.plot_hist(self.hist)

    def save(self, path):
        torch.save(self.state_dict(), path + '.pth')
        torch.save(self.hist, path + '_hist.pth')

    def load(self, path):
        self.load_state_dict(torch.load(path + '.pth', map_location='cpu'))
        self.hist = torch.load(path + '_hist.pth')
