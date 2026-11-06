import matplotlib.pyplot as plt


plt.ion()


def plot(scores, mean_scores):
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean score")
    plt.ylim(ymin=0)
    plt.legend()
    plt.pause(0.1)
