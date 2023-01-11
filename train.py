import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from adamp import AdamP

loss_fn = nn.CrossEntropyLoss(reduction='sum')


def train(model, args, train_loader, val_loader):
    optimizer = AdamP(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)
    tl = []
    vl = []
    ta = []
    va = []

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        train_loss = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = tuple(t for t in batch)
            b_text, b_label = batch
            out = model(b_text.squeeze(1))
            loss = loss_fn(out, b_label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        scheduler.step()
        avg_train_loss = np.mean(train_loss)
        avg_train_acc, _ = predict(model, train_loader)
        avg_val_acc, avg_val_loss = predict(model, val_loader)
        end = time.time()
        if epoch == 0:
            print(f"{end - start:.1f} sec")
        print("Epoch {0},  Training loss: {1:.4f},  Training accuracy: {2:.4f},  "
              "Validation loss: {3:.4f},  Validation accuracy: {4:.4f}"
              .format(epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))
        tl.append(avg_train_loss)
        vl.append(avg_val_loss)
        ta.append(avg_train_acc)
        va.append(avg_val_acc)

    plt.plot(np.arange(args.epochs), tl)
    plt.plot(np.arange(args.epochs), vl)
    plt.xticks(np.arange(0, args.epochs, 2))
    plt.title('RNN w pretrained embedding')
    plt.legend(('train_loss', 'test_loss'))
    plt.show()
    plt.plot(np.arange(args.epochs), ta)
    plt.plot(np.arange(args.epochs), va)
    plt.xticks(np.arange(0, args.epochs, 2))
    plt.title('RNN w pretrained embedding')
    plt.legend(('train_acc', 'test_acc'))
    plt.show()


def predict(model, dataloader):
    model.eval()
    prediction = []
    answer = []
    val_loss = []

    for step, batch in enumerate(dataloader):
        batch = tuple(t for t in batch)
        b_text, b_label = batch
        with torch.no_grad():
            out = model(b_text.squeeze(1))
            loss = loss_fn(out, b_label)
            val_loss.append(loss.item())
        prediction.append(out)
        answer.append(b_label)

    prediction = np.argmax(torch.cat(prediction).numpy(), axis=1)
    answer = torch.cat(answer).numpy()
    return sum(prediction == answer) / len(answer), np.mean(val_loss)
