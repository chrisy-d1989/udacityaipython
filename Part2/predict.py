#Script for flower prediction with trained neural model

from processing_functions import (get_input_args, catToName, process_image, predict, printing_results)
from model_functions import (load_checkpoint, load_model, define_classifier, set_data_dir, load_data)

def main():
    #define image path
    image_path = 'flowers/train/1/image_06734.jpg'
    
    #get inputs from the user for training checkpoint direction, architecture, top_k and mapping of categories
    input_args = get_input_args()
    # define image directions
    train_dir, valid_dir, test_dir = set_data_dir('flowers')

    #load and transpose images
    trainloader,validationloader, testloader, image_datasets_training = load_data(train_dir, valid_dir, test_dir)
  
    #load network for prediction
    model = load_model(input_args.arch)
    
    #define classifier and params
    classifier, param = define_classifier(model, input_args.hidden_units, input_args.arch)

    #load trained state_dict and epochs
    epochs, model.load_state_dict, model.class_to_idx = load_checkpoint(input_args.dir, model)
    
    #cat_to_name
    cat_to_name = catToName('cat_to_name.json')
    
    #image preprocessing
    image = process_image(image_path)
    
    # prediction of flower
    top_probs, cat_to_name_topk = predict(image_path, model, input_args.topk, image_datasets_training, cat_to_name, input_args.gpu)
    
    # printing results
    printing_results(top_probs, cat_to_name_topk)

    # Call to main function to run the program
if __name__ == "__main__":
    main()  