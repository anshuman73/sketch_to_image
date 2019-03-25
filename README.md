# sketch_to_image
Using CGAN we have generated Realistic Faces from sketches, facades to buildings etc.


## How to Use

* Get datasets
* Copy the train-model.py in the directory of the data-set that you want to train.
* Run the train-model.py after installing all the required dependencies.
* Copy the final .h5 files generated in the webapp/models directory, named using the following nomenclature - 
	* facades2buildings_discriminator.h5
	* facades2buildings_generator.h5
	* maps2buildings_discriminator.h5
	* maps2buildings_generator.h5
	* scapes2city_discriminator.h5
	* scapes2city_generator.h5
	* sketch2faces_discriminator.h5
	* sketch2faces_generator.h5
* Run the webapp/app.py file to start the flask app to run your models and provide input and generate output using a GUI.


## Description of the included files

* sketch.py was used to convert images to sketches to augment the training data.
* merge.py was used to merge the ground truth and the input image, as expected by the Neural Network.