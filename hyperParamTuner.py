import binEngBERT

epochs = [2, 3, 4, 5, 6, 7, 8]
learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
batch_sizes = [16, 32, 48, 64, 80, 96]
epsilons = [1e-8, 1e-6]
betas = [(0.9, 0.999), (0.9, 0.98)]


class BestModelParams:
    def __init__(self):
        self.e = 0
        self.lr = 0
        self.bs = 0
        self.eps = 0
        self.b = 0


global_loss = 0
global_acc = 0

bmp = BestModelParams()

# Nested epochs and learning rates, so this will take way more time than the other checks
for epoch in epochs:
    loss, acc = binEngBERT.main(epoch)

    if acc >= global_acc:
        global_acc = acc
        bmp.e = epoch
        print("Saved!")

    for learning_rate in learning_rates:
        loss, acc = binEngBERT.main(bmp.e, learning_rate)

        if acc >= global_acc:
            global_acc = acc
            bmp.lr = learning_rate
            print("Saved!")

for batch_size in batch_sizes:
    loss, acc = binEngBERT.main(bmp.e, bmp.lr, batch_size)

    if acc >= global_acc:
        global_acc = acc
        bmp.bs = batch_size
        print("Saved!")

for epsilon in epsilons:
    loss, acc = binEngBERT.main(bmp.e, bmp.lr, bmp.bs, epsilon)

    if acc >= global_acc:
        global_acc = acc
        bmp.eps = epsilon
        print("Saved!")

for beta in betas:
    loss, acc = binEngBERT.main(bmp.e, bmp.lr, bmp.bs, bmp.eps, beta)

    if acc >= global_acc:
        global_acc = acc
        bmp.beta = beta
        print("Saved!")

print("Best hyperparameters:")
print("-" * 70)
print("epochs:        ", bmp.e)
print("learning rate: ", bmp.lr)
print("batch size:    ", bmp.bs)
print("epsilon:       ", bmp.eps)
print("betas:         ", bmp.b)
