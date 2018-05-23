# script for training a neural network

from processing_functions import (get_input_args)
from model_functions import (set_data_dir, load_data, load_model, define_classifier, training_network, save_checkpoint)

def main():
    # get inputs from the user for training checkpoint direction, architecture, learning rate, hidden units and epochs
    input_args = get_input_args()

    # define image directions
    train_dir, valid_dir, test_dir = set_data_dir('flowers')

    #load and transpose images
    trainloader,validationloader, testloader, image_datasets_training = load_data(train_dir, valid_dir, test_dir)

    #load model from pytorch
    model = load_model(input_args.arch)

    #define the classifier and freeze params
    classifier, param = define_classifier(model, input_args.hidden_units, input_args.arch)

    #train the network
    model, epochs, optimizer = training_network(model,classifier,trainloader,validationloader, input_args.gpu, input_args.epochs)
    model.class_to_idx = image_datasets_training.class_to_idx

    #save network to checkpoint
    save_checkpoint(model.class_to_idx, epochs, model, optimizer, input_args.dir)

# Call to main function to run the program
if __name__ == "__main__":
    main()  