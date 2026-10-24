from torchvision.datasets import OxfordIIITPet

ROOT = "./data"

trainval_set = OxfordIIITPet(root=ROOT, split="trainval", download=True)
test_set     = OxfordIIITPet(root=ROOT, split="test", download=True)

print(len(trainval_set), len(test_set))

img, label = trainval_set[0]
print(type(img), img.size)
print(type(label), label)