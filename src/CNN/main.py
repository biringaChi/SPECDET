from CNN.test import Test
from CNN.train import Train

train = Train()
test = Test()

def main():
	train.train()
	train.log_metrics()
	test.test_accuracy()
	test.evaluate()

if __name__ == "__main__":
	main()