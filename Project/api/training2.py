#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS=3
EPOCHS=50


# In[3]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "potato datset",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[4]:


class_names=dataset.class_names
class_names


# In[5]:


len(dataset)


# In[6]:


for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())
    print(image_batch[0].numpy())


# In[7]:


plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
   for i in range(12):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")


# In[8]:


len(dataset)


# In[9]:


# 80% for training
# 20% --> 10% for validation and 10% for testing


# In[10]:


train_size=0.7
len(dataset)*train_size


# In[11]:


train_ds=dataset.take(47)
len(train_ds)


# In[12]:


test_ds=dataset.skip(47)
len(test_ds)


# In[13]:


val_size=0.15
len(dataset)*val_size


# In[14]:


val_ds=test_ds.take(10)
len(val_ds)


# In[15]:


test_ds=test_ds.skip(10)
len(test_ds)


# In[16]:


def get_dataset_partition_tf(ds, train_split=0.7, val_split=0.20, test_split=0.10, shuffle=True, shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)

    train_ds=ds.take(train_size)

    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds


# In[17]:


train_ds,val_ds,test_ds=get_dataset_partition_tf(dataset)


# In[18]:


len(train_ds)


# In[19]:


len(val_ds)


# In[20]:


len(test_ds)


# In[21]:


train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[22]:


resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.Rescaling(1.0/255)
])


# In[23]:


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])


# In[24]:


# Self-Attention Layer
class SelfAttention(layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.W1 = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.W2 = self.add_weight(shape=(self.units, 1), initializer="random_normal", trainable=True)

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W1))
        attention_weights = tf.nn.softmax(tf.matmul(score, self.W2), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# Define Model
n_class = 3
model = models.Sequential([
    layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),  # Explicitly define the input shape
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the output before passing to Self-Attention
    layers.Flatten(),
    
    # Adding Self-Attention Layer
    SelfAttention(64),
    
    # Reshape the output to ensure compatibility with Dense layers
    layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),  # Adjust output dimensions for Dense layers
    
    layers.Dense(64, activation='relu'),
    layers.Dense(n_class, activation='softmax'),
])

# Model Summary
model.summary()


# In[25]:


# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# # Train the model
# history = model.fit(
#     train_ds,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     verbose=1,
#     validation_data=val_ds
# )

# Evaluate the model
scores = model.evaluate(test_ds)


# In[26]:


scores=model.evaluate(test_ds)


# In[27]:


scores


# In[28]:


history


# In[29]:


history.params


# In[30]:


history.history.keys()


# In[31]:


history.history['accuracy']


# In[32]:


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']


# In[33]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),acc, label='Training Accuraccy')
plt.plot(range(EPOCHS),val_acc, label='Validation Accuraccy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(EPOCHS),loss, label='Training Loss')
plt.plot(range(EPOCHS),val_loss, label='Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')


# In[34]:


for images_batch, labels_batch in test_ds.take(1):
    first_image=images_batch[0].numpy().astype('uint8')
    first_label=labels_batch[0].numpy()
    print("first image to predict")
    plt.imshow(first_image)
    print("first image actual label : ",class_names[first_label])

    batch_prediction=model.predict(images_batch)
    print(batch_prediction[0])
    # predi_class_idx=np.argmax(batch_prediction[0])
    print(np.argmax(batch_prediction[0]))
    # print("predicted class : ",class_names[predi_class_idx])
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])


# In[35]:


for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    print("First image to predict:")
    plt.imshow(first_image)
    plt.show()
    print("First image actual label:", class_names[first_label])

    # Predict using the correct variable name
    batch_prediction = model.predict(images_batch)
    
    print("Prediction probabilities:", batch_prediction[0])
    predicted_class_idx = np.argmax(batch_prediction[0])
    print("Predicted class index:", predicted_class_idx)
    print("Predicted label:", class_names[predicted_class_idx])


# In[36]:


def predict(model, img):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array,0)

    predictions=model.predict(img_array)

    predicted_class=class_names[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence


# In[37]:


plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))

        predicted_class, confidence =predict(model, images[i].numpy())
        actual_class=class_names[labels[i]]

        plt.title(f"Actual : {actual_class},\n Predicted: {predicted_class}, \n Confidence: {confidence}%")
        plt.axis('off')


# In[38]:


import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to predict and return the predicted class, confidence
def predict(model, image):
    image = np.expand_dims(image, axis=0)  # Adding batch dimension
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # Confidence in percentage
    return predicted_class, confidence

# Function to calculate PSNR
def calculate_psnr(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Initialize arrays for actual and predicted labels
y_true = []
y_pred = []

# Your provided code: Loop through test dataset and plot images
plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))

        # Predict and get the class and confidence
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = labels[i].numpy()

        # Append true and predicted classes for later metrics calculation
        y_true.append(actual_class)
        y_pred.append(predicted_class)

        # Plot title with actual and predicted class along with confidence
        plt.title(f"Actual: {class_names[actual_class]},\nPredicted: {class_names[predicted_class]},\nConfidence: {confidence:.2f}%")
        plt.axis('off')

# Convert to numpy arrays for metrics calculation
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Handle dynamic class names based on y_true
unique_labels = np.unique(y_true)  # Get unique labels in y_true
target_class_names = [class_names[i] for i in unique_labels]  # Match only the relevant class names

# 2. Classification Report (Precision, Recall, F1 Score per class)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_class_names, labels=unique_labels))

# 3. Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 4. Precision, Recall, F1 Score (Weighted Average)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 5. Mean Squared Error (Optional for regression tasks)
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# 6. PSNR (Peak Signal-to-Noise Ratio) for image comparison
# Assuming you're comparing actual vs predicted image labels
psnr_values = [calculate_psnr(images[i].numpy(), images[i].numpy()) for i in range(9)]  # Placeholder comparison
print(f"PSNR: {np.mean(psnr_values):.2f} dB")

# 7. AUC (only for binary classification tasks)
if len(unique_labels) == 2:  # AUC is for binary classification
    auc = roc_auc_score(y_true, y_pred)
    print(f"AUC: {auc:.4f}")
else:
    print("AUC not applicable for multi-class classification.")


# In[ ]:





# In[39]:


import tensorflow as tf
print(tf.__version__)


# In[40]:


import tensorflow as tf
import os

model_version = 100
model_save_path = os.path.join("..", "models", f"model_v{model_version}.keras")

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the model
model.save(model_save_path)



