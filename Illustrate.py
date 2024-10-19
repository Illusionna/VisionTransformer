import os
import json
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
path = './cache/Fruits-train.json'
# ---------------------------------------------------------------

with open(path, mode='r', encoding='utf-8') as f:
    data = json.load(f)
epoch = data['epoch']
train_accuracy = data['train_accuracy']
train_loss = data['train_loss']
valid_accuracy = data['valid_accuracy']
valid_loss = data['valid_loss']
plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = 14
plt.plot(epoch, train_accuracy, linewidth=3, color=(240/256, 50/256, 120/256), label='train accuracy')
plt.plot(epoch, train_loss, linewidth=3, color=(240/256, 120/256, 50/256), label='train loss')
plt.plot(epoch, valid_accuracy, linewidth=3, color=(0/256, 120/256, 185/256), label='valid accuracy')
plt.plot(epoch, valid_loss, linewidth=3, color=(50/256, 185/256, 240/256), label='valid loss')
plt.xticks([i for i in range(0, len(epoch)+10, 10)])
plt.xlabel('Epoch')
plt.ylabel('Result')
plt.grid(which='major', linestyle='-.', alpha=0.33, color='black')
plt.legend(loc='lower left')
plt.title(os.path.basename(path))
plt.show()