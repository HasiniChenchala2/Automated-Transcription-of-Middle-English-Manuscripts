import xml.etree.ElementTree as ET

#https://docs.python.org/3/library/xml.etree.elementtree.html

def xmltotxt():
    root = ET.parse("FWhole.xml")
    # print(root)
    # Function to extract text from an element
    def extract_text(element):
        if element.tag != 'note':
            text = element.text or ""
            for sub_element in element:
                if sub_element.tag == 'expan':
                    text += "{" + extract_text(sub_element) + "}"
                else:
                    text += extract_text(sub_element) 
                # print(sub_element.tail)
                text += sub_element.tail or ""
            if element.tag =='l':
                text += "\n"
            return text
        return ""
    with open('output.txt', 'w',encoding="utf-8") as f:
        for lg in (root.findall(".//lg")):
            lg_text = extract_text(lg) + "\n\n"
            f.write(lg_text)
    
            
    
if __name__ == '__main__':

    xmltotxt()

