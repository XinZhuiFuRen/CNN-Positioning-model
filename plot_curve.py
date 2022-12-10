import pandas as pd
import matplotlib.pyplot as plt

logs = pd.read_csv('train.log')

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.plot(logs['loss'], label='train')
plt.plot(logs['test_loss'], label='test')
plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(2, 2, 2)
plt.plot(logs['acc1'], label='train')
plt.plot(logs['test_acc1'], label='test')
plt.legend()
plt.ylabel('acc1')
plt.xlabel('epoch')

plt.subplot(2, 2, 3)
plt.plot(logs['acc2'], label='train')
plt.plot(logs['test_acc2'], label='test')
plt.legend()
plt.ylabel('acc2')
plt.xlabel('epoch')

plt.subplot(2, 2, 4)
plt.plot(logs['acc3'], label='train')
plt.plot(logs['test_acc3'], label='test')
plt.legend()
plt.ylabel('acc3')
plt.xlabel('epoch')

plt.tight_layout()
plt.show()