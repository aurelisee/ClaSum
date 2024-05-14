from transformers import BartTokenizer, BartForConditionalGeneration

def prepend_text_to_file(file_path, text_to_prepend):
    try:
        # Step 1: Read existing content from the file
        with open(file_path, 'r') as file:
            existing_content = file.read()

        # Step 2: Prepare new content by combining text_to_prepend and existing content
        new_content = text_to_prepend + existing_content

        # Step 3: Write the combined content back to the file
        with open(file_path, 'w') as file:
            file.write(new_content)

        print("Text prepended successfully.")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_summary(text, j):
    # Load pre-trained BART-Large tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    text_length = len(text.split())
    desired_summary_length = int(text_length * 0.80)
    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    print(len(inputs))
    # Generate the summary
    summary_ids = model.generate(inputs.input_ids, num_beams=16, min_length=desired_summary_length, max_length=desired_summary_length+50, early_stopping=True)#(inputs.input_ids, num_beams=4, min_length=150, max_length=300, early_stopping=True)#min_length=int(len(inputs.input_ids)*0.3), max_length=300, early_stopping=True) (inputs.input_ids, num_beams=4, min_length=50, max_length=200, early_stopping=True)#

    # Decode the summary tokens
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    #with open('final_summaries\\'+str(j)+'.txt', 'w', encoding='utf-8') as d:
    #  d.write(summary)
    return summary

if __name__ == "__main__":
    for i in range(1,36):       #33
        classified_sent_path = "classifiedsentences\\"+str(i)+".txt"
        file = open('2_sentence_ranking\\ranked_sentences\\'+str(i)+'.txt', "r", encoding="UTF-8")
        FileContent = file.read().strip()  
        summary = generate_summary(FileContent, i)
        prepend_text_to_file(classified_sent_path, summary)

    
    
#C:\Users\user\Desktop\SDSADS_modif
#C:\\Users\\user\\Downloads\\BugSum-master2\\BugSum-master\\SOModelData\\final_summaries\\bart_summaries\\bartorg\\after1_just500\\
#C:\\Users\\user\\Downloads\\BugSum-master2\\BugSum-master\\SOModelData\\final_summaries\\after1part_just\\