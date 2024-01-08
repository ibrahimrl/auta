from telegram import Update
from telegram.ext import Updater, MessageHandler, CallbackContext

import requests
import uuid, time, os 

from telegram import Update

from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

def predict(dst_file_path, random_id):
    return 'Mercedes E' 

async def handle_image(update: Update, context: CallbackContext):

    print("Inside handle_image  ")
    # Check if the message contains a photo
    if update.message.photo:

        print("inside the update.message.photo")
        # Get the file ID of the largest version of the photo
        file_id = update.message.photo[-1].file_id
        print("file_id", file_id)

        
        # Get the file object from the file ID
        file = await context.bot.get_file(file_id)

        print(file.file_path)

        response = requests.get(file.file_path)
        
        random_id = uuid.uuid4()

        path = f'{time.strftime("%H:%M:%S")}_{str(random_id)}.jpg' #os.path.join('data/croped', )
        directory = 'data/input_images'
        
        if not os.path.exists(directory):
        # Create the directory if it doesn't exist
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")
        
        
        dst_file_path = os.path.join(directory, path)

        with open(dst_file_path, 'wb') as f:
            f.write(response.content)

        await update.message.reply_text('Processing....' )

        make_model = predict(dst_file_path, random_id)

        
        await update.message.reply_text(f'Recognized Make Model : {make_model}')

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Reply")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Salam")



def main():

    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    message_handler = MessageHandler(filters.PHOTO, handle_image)
    application.add_handler(message_handler)



    message_handler2 = MessageHandler(filters.TEXT & (~filters.COMMAND), reply)
    application.add_handler(message_handler2)
    

    print("Starting ......")

    application.run_polling()


if __name__ == '__main__':
    main()