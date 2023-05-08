import React from 'react'

const Architecture = () => {
  return (
    <div>
      <div className="flex justify-center">
        <h1 className="text-4xl font-bold text-center text-gray-800">
          Architecture
        </h1>
      </div>
      <div className="flex flex-col items-center justify-center ">
        <div className="max-w-3xl">
          <img className="mx-auto rounded-md shadow-lg" src="ML_Component_Architecture.png" alt="Hateful Meme" />
        </div>
        <p className="text-left mt-4 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          The figure above shows a high level architecture that has been used in this project. This architecture is
          specifically designed for training purposes and is followed rigorously to make the process of building model
          smoother. The architecture is divided into three different modules. First: A Visual Processing Unit (VPU),
          Second: A Textual Processing Unit (TPU) and Third: Fusion model. Please note that even though the process seems
          streamlined, it is not.
        </p>
      </div>

      <div className="flex justify-center mt-4">
        <h1 className="text-4xl font-bold text-center text-gray-800">
          Visual Processing Flow (VPU)
        </h1>
      </div>

      <div className="flex flex-col items-center justify-center mt-4">
        <p className="text-left max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          The visual processing unit (VPU) is responsible to extract the visual features from the batch of images that is
          provided to it while training. To extract the features, first the images are preprocessed. The preprocessing
          includes three steps viz resizing, augmentation and transforms. Resizing step includes to make the dimensions
          of images conform to (224, 224, 3) which is the most preferred dimension for any computer vision algorithm.
          Augmentation includes randomly performing either rotation, addition of gaussian noise and addition of blur to
          each image. As a result, the dataset becomes richer and helps the model to learn about various noise that may
          come while performing inference. After augmenting the images, applying transformation to images is essential.
          This transformation requires to normalize the images with a very specific value of mean and standard deviation
          for all the three layers (RGB layers). This value (Mean = [0.485, 0.456, 0.406] and SD = [0.229, 0.224, 0.225])
          is universally accepted value which has proved to improve the model performance on numerous occasions.
        </p>
      </div>

      <div className="flex justify-center mt-4">
        <h1 className="text-4xl font-bold text-center text-gray-800">
          Textual Processing Flow (TPU)
        </h1>
      </div>

      <div className="flex flex-col items-center justify-center mt-4">
        <p className="text-left max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          Once all the images have been preprocessed, the batch of the dataset is ready to be passed through the model.
          For the model, initially ResNet50 was used which was replaced by YOLOv8 in the hope of improving the performance
          of the model. ResNet50 is a 50 layer architecture containing multiple blocks of CNN, pooling and activated layers.
          At the end, ResNet50 model provides the features of the dimension (BATCH\_SIZE, 2048). Since the number of neurons
          are very high, we need additional layers to bring the number of features down. Thus, two additional layers are
          with ReLU activation function bringing the dimension of the output features to (BATCH\_SIZE, 512).
        </p>
      </div>

      <div className="flex justify-center mt-4">
        <h1 className="text-4xl font-bold text-center text-gray-800">
          Fusion Layer
        </h1>
      </div>

      <div className="flex flex-col items-center justify-center mt-4 mb-4">
        <p className="text-left max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          Finally, the third component of the architecture is the fusion layer(s). This layer is the most essential
          component of the entire project. This component gets the extracted visual and textual features from the
          respective units and concatenated them. Since the concatenation happens after the features were extracted,
          it is a late fusion model. In addition, the concatenated features are passed to additional layers that understand
          the combined meaning of the image and text to classify the meme as either hateful or non-hateful.
        </p>
      </div>
    </div>
  )
}

export default Architecture