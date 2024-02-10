# Car Make and Model Recognition Telegram Bot

This project consists of a Telegram bot that predicts the make and model of cars from images sent via the Telegram app. The bot utilizes a custom-trained model to perform the recognition.

## Project Structure

The project is organized as follows:

1. **Folder_Reordering**: Contains scripts for preprocessing the data collected from a car selling website. This includes organizing images into folders based on make and model, and merging similar folders.
    - **copy_subfolders.py**: Python script to copy subfolders into a single folder.
    - **single_image_deleter.py**: Python script to delete subfolders containing only a single image.
    - **merge_folders.py**: Python script to merge similar folders.

2. **Customized Detect**: Customized version of YOLOv5's detect.py file, modified to detect all vehicles in an image and save the largest vehicle. Includes an option to anonymize license plates.
    - **customized_detect.py**: Modified version of YOLOv5's detect.py file.
    - **weights/LP_Detect_weight.pt**: Pre-trained weights for license plate detection.

3. **MakeModel_Rec**: Contains scripts and files related to training and using the car make and model recognition model.
    - **train_model.py**: Python script for training the car make and model recognition model.
    - **weights/MakeModel_Rec.pt**: Trained weights for the make and model recognition model.
    - **Class_Names.txt**: Text file containing human-readable class names for the model's predictions.

4. **telegram_server.py**: Python script for the Telegram bot server. Receives images from users, predicts the car make and model, and sends the results back.



## Usage

To use the Telegram bot:

1. Start the `telegram_server.py` script.
2. Send an image of a car to the Telegram bot.
3. Receive the predicted make and model of the car as a response.

## Credits

- License plate detection model based on YOLOv5 by [KALYAN1045](https://github.com/KALYAN1045/Automatic-Number-Plate-Recognition-using-YOLOv5).
- Initial data collection and preprocessing inspired by [car selling website](https://www.example.com).
- Model training and development by [Your Name or Team Name].

## License

This project is licensed under the [MIT License](LICENSE).
