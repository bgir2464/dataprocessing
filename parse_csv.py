from pandas import read_excel

def get_pictures_urls(file_name):
    my_sheet = 'results' # change it to your sheet name, you can find your sheet name at the bottom left of your excel file

    df = read_excel(file_name,sheet_name = my_sheet, usecols = "I,K")
    urls = df.values.tolist()
    urls_f=[x[0] for x in urls if x[0]!='nan' and  x[0]!='URL']   #if x[1]=="Mimbres Pottery Images Digital Database with Search"]

    return urls_f



