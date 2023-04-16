# CelebA attributes prediction in a multi-task Fashion
This assigment is an analysis of the CelebA dataset, which consists of over 200,000 celebrity images with annotations for facial landmarks, attributes, and identities.
# Prerequisites
Numpy<br>
Pandas<br>
Pytorch<br>
VGG16<br>
### Download the dataset directly from Pytorch
```datasets.CelebA(root='.', split='train', transform=transform, target_type='attr', download=True)```
### VGG16
The VGG16 architecture consists of 16 layers, including 13 convolutional layers and 3 fully connected layers. The convolutional layers use small 3x3 filters, which allow for a deeper network with a smaller number of parameters compared to other architectures. The network is trained on the ImageNet dataset, which consists of over 1 million images with 1000 categories, and uses a cross-entropy loss function.
### Load the pre-trained VGG-16 model on Imagenet
```vgg16 = models.vgg16(pretrained=True)```
<br>
I added some fully connected layer at the end of VGG16<br>
The model is trained for 5 epochs as the time for running each epoch is >20 mins on GPU.<br>
Task wise accuracies and overall accuracies are shown in the python file.

## Dynamic schedule Task (DST)
 While training a deep network, the model may overfit if the training samples are low and trained for a long duration (higher number of epochs). On the contrary, the network may underfit if it is trained for lesser epochs with larger number of training samples. For Multi Task Learning, not all tasks may have an equal number of ground-truth labels. The task with fewer annotated samples might benefit (positive transfer) from the large corpus of samples provided by another task with more labeled samples. However, a target task with more labeled samples might undergo a negative transfer due to a source task with fewer instances. Hence, each task requires computing cycles proportional to the number of labeled instances.
``` y=y[:,:8]
       y=y.reshape(8,-1)
       gt_counts=[0,0,0,0,0,0,0,0]
       for i in range(8):
          for j in range(64):
            if(y[i,j]==1):
               gt_counts[i]+=1
       sum_values = sum(gt_counts)
       normalized = [x / sum_values for x in gt_counts]
       weights = [int(random.random() < normalized[t]) for t in range(8)]
```     
The above code block is run at each epoch and the resulting weights which are [0,1] are multiplied to the loss function of the individual tasks resulting in ON/OFF scenario in training of the task.Here, the range is 8 as we are training only 8 out of 40 attributes.<br>
This resulted in significant increase in overall accuracies.

