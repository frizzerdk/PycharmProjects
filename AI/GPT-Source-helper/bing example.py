import json
import os
import openai
import re
from pyzotero import zotero
from pdfminer.high_level import extract_text


def ask_gpt35(system_intel, prompt):
    offline= False
    if offline:
        output='{"itemType": "journalArticle", "title": "", "creators": [{"creatorType": "author", "firstName": "", "lastName": ""}], "abstractNote": "", "publicationTitle": "", "volume": "", "issue": "", "pages": "", "date": "", "series": "", "seriesTitle": "", "seriesText": "", "journalAbbreviation": "", "language": "", "DOI": "", "ISSN": "", "shortTitle": "", "url": "", "accessDate": "", "archive": "", "archiveLocation": "", "libraryCatalog": "", "callNumber": "", "rights": "", "extra": "", "tags": [], "collections": [], "relations": {}}'
        output ='{"itemType": "journalArticle", "title": "AutoML: A Survey of the State-of-the-Art", "creators": [{"creatorType": "author", "firstName": "Xin", "lastName": "He"}, {"creatorType": "author", "firstName": "Kaiyong", "lastName": "Zhao"}, {"creatorType": "author", "firstName": "Xiaowen", "lastName": "Chu"}], "abstractNote": "Deep learning (DL) techniques have obtained remarkable achievements on various tasks, such as image recognition, object detection, and language modeling. However, building a high-quality DL system for a speciﬁc task highly relies on human expertise, hindering its wide application. Meanwhile, automated machine learning (AutoML) is a promising solution for building a DL system without human assistance and is being extensively studied. This paper presents a comprehensive and up-to-date review of the state-of-the-art (SOTA) in AutoML. According to the DL pipeline, we introduce AutoML methods –– covering data preparation, feature engineering, hyperparameter optimization, and neural architecture search (NAS) –– with a particular focus on NAS, as it is currently a hot sub-topic of AutoML. We summarize the representative NAS algorithms’ performance on the CIFAR-10 and ImageNet datasets and further discuss the following subjects of NAS methods: one/two-stage NAS, one-shot NAS, joint hyperparameter and architecture optimization, and resource-aware NAS. Finally, we discuss some open problems related to the existing AutoML methods for future research. Keywords: deep learning, automated machine learning (AutoML), neural architecture search (NAS), hyperparameter optimization (HPO)", "publicationTitle": "Knowledge-Based Systems", "volume": "", "issue": "", "pages": "", "date": "2021", "series": "", "seriesTitle": "", "seriesText": "", "journalAbbreviation": "", "language": "English", "DOI": "", "ISSN": "", "shortTitle": "", "url": "", "accessDate": "", "archive": "", "archiveLocation": "", "libraryCatalog": "", "callNumber": "", "rights": "", "extra": "", "tags": ["AutoML", "State-of-the-Art", "Deep Learning", "Neural Architecture Search", "Hyperparameter Optimization"], "collections": [], "relations": {}}'

    else:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": system_intel},
                {"role": "user", "content": prompt},
            ],d
            temperature=0.01)
        output = result['choices'][0]['message']['content']
    return output


def create_zotero_item(system_intel, prompt, zot):
    response = ask_gpt35(system_intel, prompt)
    print('response', response)

    if response is not None:
        returned_template = response[response.find("{"):response.rfind("}") + 1]
        returned_template = returned_template.replace("'", "\"")
       # returned_template = json.dumps(zot.item_template("journalArticle"))
        template_json = json.loads(returned_template)


        item = zot.item_template("journalArticle")

        for key, value in template_json.items():
            item[key] = value

        return item
    else:
        return None





def process_pdf_files(pdfs, zot, system_intel):
    for filename in os.listdir(pdfs):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdfs, filename)
            text = extract_text(filepath)
            words = text.split()
            selected_words = words[:2000]
            start_of_text = ' '.join(selected_words)

            print('Extracted Text:\n', start_of_text)

            template = zot.item_template("journalArticle")
            if not template['tags']:
                # add a dummy tag object with an empty tag property
                template['tags'].append({'tag': ''})
            template_s = json.dumps(template)
            prompt = f"Given this opening text from a PDF document that is a paper:\n\n{start_of_text}\n\n help me create a Zotero item for this paper by filling in and returning only this template with no additional text before or after with any relevant fields based on the text. Stick with the fields of the template, this is the template : \n\n{template_s}"

            print('prompt\n', prompt)

            item = create_zotero_item(system_intel, prompt, zot)
            # Create the item in Zotero library and get its 'key'
            created_item = zot.create_items([item])
            print('created_item\n', created_item)
            item_key = created_item['successful']['0']['key']

            if item is not None:
                # Create attachment item
                attachment_item_data = zot.attachment_simple([filepath], parentid=item_key)

                # Check if the attachment item was created successfully
                if attachment_item_data['success']:
                    # Upload the file to the attachment item
                    with open(filepath, 'rb') as pdf_file:
                        attachment_info = [{'filename': filepath}]
                        zot.upload_attachments(attachment_info, parentid=item_key)

                    print(f"Created Zotero item and attached file for {filename}")
                else:
                    print(f"Failed to create attachment item for {filename}")

            else:
                print(f"OpenAI request failed for {filename}")
        else:
            print(f"{filename} is not a PDF file")
def main():
    pdfs = "C:/Users/frizzer/Google Drive/Papers/AI/"
    library_id = "11358086"
    library_type = "user"
    api_key = "B2l2Ko9QIJT0XJ3PWIHJWJAC"
    openai_key = "sk-"

    openai.api_key = openai_key
    zot = zotero.Zotero(library_id, library_type, api_key)

    system_intel = "You are an expert in research papers and citations and helping with making entries for a zotero library using the pyzotero api, answer my questions as if you were an expert in the field."

    process_pdf_files(pdfs, zot, system_intel)

if __name__ == "__main__":
    main()
