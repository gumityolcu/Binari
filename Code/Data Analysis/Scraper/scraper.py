from selenium import webdriver
import os


def get_poems():
    driver = webdriver.Chrome(os.path.join(os.getcwd(),"chromedriver"))
    driver.get("https://safahat.diyanet.gov.tr/PoemList.aspx?all=1l")
    poemList = driver.find_element_by_xpath('/html/body/form/div[4]/div[1]/div[3]')
    links = poemList.find_elements_by_tag_name("a")
    links = [l.get_attribute("href") for l in links]
    i=0
    allParagraphs = list()
    for l in links:
        driver.get(l)
        try:
            paragraphs = driver.find_elements_by_class_name("siir")[0].find_elements_by_tag_name("p")
            allParagraphs += [p.text for p in paragraphs]
        except Exception as e:
            print(e)
    f =open("safahat.txt","w")
    for p in allParagraphs:
        if p[-1]!="\n":
            p+="\n"
        f.write(p)
    f.close()
