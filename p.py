import numpy as np
import time
import os
import matplotlib.pyplot as plt

#Data Loading
def load_images(filepath, width, height):
    images = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), height):
        block = lines[i:i+height]
        img = []
        for row in block:
            row = row.rstrip('\n')
            row = row.ljust(width)          # pad short lines
            for ch in row[:width]:
                # Dataset encoding: ' ' and '+' are background (0); '#' is ink/edge (1)
                img.append(0 if (ch == ' ') else 1)
        if len(img) == width * height:
            images.append(np.array(img, dtype=np.float32))
    return images

def load_labels(filepath):
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

#Perceptron Class with hard threshold prediction and mistake-driven updates.
class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=10):
        # Initialize weights and bias
        self.w = np.zeros(input_size, dtype=np.float32)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def score(self, x):
        return float(np.dot(self.w, x) + self.b)

    def predict(self, x):
        return 1 if self.score(x) >= 0.0 else 0

    def train(self, X, y):
        X_list = list(X)
        y_list = list(y)
        for _ in range(self.epochs):
            order = np.random.permutation(len(X_list))
            for i in order:
                xi = X_list[i]
                yi = int(y_list[i])
                y_pred = self.predict(xi)
                if y_pred != yi:
                    delta = (yi - y_pred)
                    self.w += self.lr * delta * xi
                    self.b += self.lr * delta

#For digits 0-9, train 10 perceptrons. Perceptron k learns "is this digit k or not?"
class OneVsAllPerceptron:
    def __init__(self, input_size, num_classes, lr=0.01, epochs=10):
        self.classifiers = [
            Perceptron(input_size, lr, epochs) for _ in range(num_classes)
        ]
        self.num_classes = num_classes

    def train(self, X, y):
        for k, clf in enumerate(self.classifiers):
            binary_labels = [1 if yi == k else 0 for yi in y]
            clf.train(X, binary_labels)

    def predict(self, x):
        # Pick class with highest raw activation score
        scores = [clf.score(x) for clf in self.classifiers]
        return int(np.argmax(scores))


#Run 5 random trials for each
def run_experiment(X_train, y_train, X_test, y_test, task='face',
                   percentages=None, n_trials=5, lr=0.01, epochs=10):
    if percentages is None:
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    input_size = X_train[0].shape[0]
    num_classes = 10 if task == 'digit' else 2
    results = []

    for pct in percentages:
        accs, times = [], []
        n = max(1, int(len(X_train) * pct))

        for _ in range(n_trials):
            # Randomly sample n training points
            idx = np.random.choice(len(X_train), n, replace=False)
            X_sub = [X_train[i] for i in idx]
            y_sub = [y_train[i] for i in idx]

            # Train
            t0 = time.time()
            if task == 'digit':
                model = OneVsAllPerceptron(input_size, num_classes, lr, epochs)
            else:
                model = Perceptron(input_size, lr, epochs)
            model.train(X_sub, y_sub)
            elapsed = time.time() - t0

            # Test
            correct = sum(model.predict(x) == y for x, y in zip(X_test, y_test))
            acc = correct / len(X_test)
            accs.append(acc)
            times.append(elapsed)

        results.append({
            'pct': int(pct * 100),
            'n': n,
            'mean_acc': np.mean(accs),
            'std_acc': np.std(accs),
            'mean_time': np.mean(times),
        })
        print(f"  {int(pct*100):3d}% | n={n:4d} | "
              f"acc={np.mean(accs):.3f} ± {np.std(accs):.3f} | "
              f"time={np.mean(times):.2f}s")

    return results

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.environ.get('PERCEPTRON_DATA_DIR', os.path.join(SCRIPT_DIR, 'data'))

    FACE_TRAIN_IMGS  = os.path.join(DATA_DIR, 'facedata', 'facedatatrain')
    FACE_TRAIN_LBLS  = os.path.join(DATA_DIR, 'facedata', 'facedatatrainlabels')
    FACE_TEST_IMGS   = os.path.join(DATA_DIR, 'facedata', 'facedatatest')
    FACE_TEST_LBLS   = os.path.join(DATA_DIR, 'facedata', 'facedatatestlabels')

    DIGIT_TRAIN_IMGS = os.path.join(DATA_DIR, 'digitdata', 'trainingimages')
    DIGIT_TRAIN_LBLS = os.path.join(DATA_DIR, 'digitdata', 'traininglabels')
    DIGIT_TEST_IMGS  = os.path.join(DATA_DIR, 'digitdata', 'testimages')
    DIGIT_TEST_LBLS  = os.path.join(DATA_DIR, 'digitdata', 'testlabels')

    FACE_W,  FACE_H  = 60, 70
    DIGIT_W, DIGIT_H = 28, 28

    print("Loading face data...")
    face_X_train = load_images(FACE_TRAIN_IMGS, FACE_W, FACE_H)
    face_y_train = load_labels(FACE_TRAIN_LBLS)
    face_X_test  = load_images(FACE_TEST_IMGS,  FACE_W, FACE_H)
    face_y_test  = load_labels(FACE_TEST_LBLS)

    print("Loading digit data...")
    digit_X_train = load_images(DIGIT_TRAIN_IMGS, DIGIT_W, DIGIT_H)
    digit_y_train = load_labels(DIGIT_TRAIN_LBLS)
    digit_X_test  = load_images(DIGIT_TEST_IMGS,  DIGIT_W, DIGIT_H)
    digit_y_test  = load_labels(DIGIT_TEST_LBLS)

    print(f"\nFace  train={len(face_X_train)}  test={len(face_X_test)}")
    print(f"Digit train={len(digit_X_train)} test={len(digit_X_test)}")

    print("\n=== FACE CLASSIFICATION (Perceptron) ===")
    face_results = run_experiment(
        face_X_train, face_y_train,
        face_X_test,  face_y_test,
        task='face', lr=0.01, epochs=15
    )

    print("\n=== DIGIT CLASSIFICATION (One-vs-All Perceptron) ===")
    digit_results = run_experiment(
        digit_X_train, digit_y_train,
        digit_X_test,  digit_y_test,
        task='digit', lr=0.01, epochs=15
    )

    pcts = [r['pct'] for r in face_results]

    plots_dir = os.path.join(SCRIPT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plt.plot(pcts, [r['mean_time'] for r in face_results], '--', label="Average Time Taken for Training Face Perceptron")
    plt.xlabel("Percent of Data Used for Training")
    plt.ylabel("Seconds")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'face_perceptron_time.png'))
    plt.show()

    plt.plot(pcts, [r['mean_time'] for r in digit_results], '--', label="Average Time Taken for Training Digit Perceptron")
    plt.xlabel("Percent of Data Used for Training")
    plt.ylabel("Seconds")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'digit_perceptron_time.png'))
    plt.show()

    plt.plot(pcts, [r['mean_acc'] for r in face_results], '--', label="Average Accuracy of Face Perceptron")
    plt.xlabel("Percent of Data Used for Training")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'face_perceptron_accuracy.png'))
    plt.show()

    plt.plot(pcts, [r['mean_acc'] for r in digit_results], '--', label="Average Accuracy of Digit Perceptron")
    plt.xlabel("Percent of Data Used for Training")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'digit_perceptron_accuracy.png'))
    plt.show()

    plt.plot(pcts, [r['std_acc'] for r in face_results], '--', label="Standard Deviation of Face Perceptron Accuracy")
    plt.xlabel("Percent of Data Used for Training")
    plt.ylabel("Stdev")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'face_perceptron_stdev.png'))
    plt.show()

    plt.plot(pcts, [r['std_acc'] for r in digit_results], '--', label="Standard Deviation of Digit Perceptron Accuracy")
    plt.xlabel("Percent of Data Used for Training")
    plt.ylabel("Stdev")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'digit_perceptron_stdev.png'))
    plt.show()