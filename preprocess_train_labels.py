text_file = open("dataset/train.txt", "r")
lines = text_file.read().split(',')
names = []
for i,pokestr in enumerate(lines,start=1):
	pokename = pokestr.split('\n')[0]
	names.append(pokename)

test_file = open("dataset/train_labels.txt","w")

for i in names:
	test_file.write(i)
	test_file.write('\n')
print names
