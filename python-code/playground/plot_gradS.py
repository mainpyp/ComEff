import matplotlib.pyplot as plt
import datetime

pis = [i*10 for i in range(1, 10)] + [95]


def equation_sparse(percentile):
    return (100 * 500 * 32 * 500000 * (1 - percentile/100)) + (500 * 500000) + (100 * 500 * 32 * 500000)


def equation_mcsu(iteration):
    return 2*100*500*500000*32/iteration


def from_bit_to_gb(bit):
    return bit / 8 / 1000 / 1000 / 1000


def from_bit_to_time(bit, mbps):
    minutes = bit / 8 / 1000 / 1000 / mbps / 60
    print(f"minutes={minutes}")
    hours = datetime.timedelta(minutes=minutes)
    return hours

traffic = [round(from_bit_to_gb(equation_sparse(percentil))) for percentil in pis]

plt.plot(pis, traffic, c="navy")
plt.grid(axis="y")
plt.title("Change of data-traffic with more local updates\n"
          "N=100 C=100 G=500.000 L=32")
plt.xticks(pis)
plt.yticks(traffic)
plt.ylabel("Data-traffic in GB")
plt.xlabel("Percentiles")


if __name__ == '__main__':
    #plt.show()
    plt.savefig("../benchmarking/percentiles.png")