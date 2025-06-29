# IMAGE-CLASSIFICATION-MODEL

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : CHINTHAPARTHI THISHITHA

*INTERN ID* : CT06DM1408

*DOMAIN* : MACHINE LEARNING

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

The image classification project presented here is a comprehensive implementation of a convolutional neural network (CNN) using TensorFlow and Keras to classify flower images into five categories: daisies, dandelions, roses, sunflowers, and tulips. The primary goal was to create a model that could recognize and accurately classify flower species based on visual features. This was achieved by leveraging TensorFlow’s high-level APIs and a publicly available dataset.

The dataset used in this project comes from TensorFlow’s official resources and is downloaded directly via a utility function. It contains around 3,700 images, structured in directories where each folder name corresponds to a flower category. These images are automatically labeled by the folder structure when loaded. After download, the data is split into training and validation sets in an 80:20 ratio using image_dataset_from_directory(), which also handles image resizing (to 180x180 pixels) and batching (batch size of 32).

To improve generalization and prevent overfitting, data augmentation techniques were applied. These include random horizontal flipping, small rotations, and zoom transformations. This simulated a more diverse dataset and enabled the model to better adapt to real-world inputs. Additionally, TensorFlow’s data pipeline optimizations such as caching, shuffling, and prefetching were implemented to ensure smoother and faster training.

The CNN model was built using the Sequential API, starting with an input normalization layer followed by three convolutional layers. Each convolutional block increases in depth (16, 32, 64 filters) and is paired with a max pooling layer to reduce dimensionality. A dropout layer with a rate of 0.2 was included to minimize overfitting, followed by a flattening layer and two dense layers—the last one representing the output classes. The model used the Adam optimizer and sparse categorical crossentropy loss function, making it suitable for multiclass classification with integer labels.

The model was trained for 15 epochs and evaluated on the validation dataset. Performance metrics such as training and validation accuracy, along with loss values, were plotted to observe the learning behavior. It was noted that while training accuracy increased steadily, validation accuracy showed signs of early saturation—common in models trained on relatively small datasets. However, the use of augmentation and dropout helped control overfitting.

Sample images from the training set were displayed using Matplotlib to ensure proper loading and labeling. Additionally, the model’s performance was tested on an external image (a red sunflower), where it successfully predicted the class with a high confidence score. This demonstrated the model’s ability to generalize beyond the training data.

The final model was also converted into TensorFlow Lite format to support deployment on edge devices like smartphones and microcontrollers. The converted model was tested using the TFLite interpreter, and the predictions closely matched those from the original Keras model, indicating successful conversion without loss of performance.

During the project, a few issues were encountered. One notable warning was about passing input_shape directly to layers like Rescaling. This was resolved by using an explicit Input() layer as the first component of the model, which aligns with TensorFlow’s best practices. Another problem was a NameError when trying to access class_names before defining it. This was fixed by ensuring the variable was declared immediately after dataset creation. Some model overfitting was also observed, which was addressed through data augmentation and dropout regularization.

In conclusion, this project provides a robust pipeline for image classification using CNNs in TensorFlow. It covers essential machine learning components—data preparation, augmentation, model architecture, training, evaluation, visualization, and deployment. With additional enhancements like confusion matrices, early stopping, or transfer learning, the project can be extended into a more advanced and production-ready system.

![Image](https://github.com/user-attachments/assets/d919b914-79bd-4281-a4c6-3a3022fa5478)

![Image](https://github.com/user-attachments/assets/fe2f2bce-2e5d-4c88-8e57-fc6b4c89f6a2)

![Image](https://github.com/user-attachments/assets/153b6e11-9ac3-46e6-b1cb-cdd863e65183)

![Image](https://github.com/user-attachments/assets/adbdad25-febc-457c-909b-5c364287cc19)

![Image](https://github.com/user-attachments/assets/cbe58b23-291b-4516-89cf-8e39eff92385)

![Image](https://github.com/user-attachments/assets/1c4cefc5-a7c9-4f80-9c2e-88a2b66892e6)

![Image](https://github.com/user-attachments/assets/0352b4e8-32b4-4d26-ba19-de0c4b6b7818)

![Image](https://github.com/user-attachments/assets/477fe707-b899-4c8f-af55-aed20de325ed)

![Image](https://github.com/user-attachments/assets/3e71a7aa-750e-4617-81e3-e696120a0adc)
