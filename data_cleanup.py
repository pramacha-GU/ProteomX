
'''
This program cleans up data to be used for the project
'''
import pandas as pd
import numpy as np

def initial_cleanup(text_cont_raw):
    '''
    This function parses text file and separates the various fields
    and puts these into a list. One for each abstract
    '''
    text_raw = text_cont_raw.split('\n')
    abstract_list = []
    text_list = []
    text_list_clean = []

    for i,j in enumerate(text_raw):
        if j == '':
            text_list_clean.append(''.join(text_list))
            text_list = []
        #print i
            if text_raw[i-1] == '':
                abstract_list.append(text_list_clean[:5])
                text_list_clean = []
        else:
            text_list.append(j)
    return abstract_list

def adding_to_txt(abstract_list):
    '''
    This function writes the data into a text file. Fields are separated by |
    '''
    file_wr = open('pubmed_cleaned.txt', 'w')
    abstract_list= abstract_list[1:]
    for num,cont in enumerate(abstract_list):
        if 'PMID:' not in cont[-1]:
            for indvl in cont:
                file_wr.write(indvl)
                file_wr.write("|")
            if num!= length_abstract-1:
                file_wr.write('\n')
    file_wr.close()

def cleaning_individual_fields():
    '''
    This function splits each field and puts them into appropriate columns in a
    DataFrame
    '''
    df = pd.read_csv('pubmed_cleaned.txt', sep='|')
    jif = pd.read_csv('Journal_Impact.csv')
    del df[6]
    del df[5]
    f3 = lambda x: x.split(';')[0]
    df1 = pd.DataFrame()
    df1['Pubmed_Index'] = np.array([x.split('.')[0] for x in df.ix[:,0]])
    df1['Journal'] = np.array([x.split('.')[1] for x in df.ix[:,0]])
    df1['Publicn_Date'] = np.array([x.split('.')[2] for x in df.ix[:,0]]
    df1['Pub_Year'] = np.array([(x.split('.')[2]).split()[0] for x in df.ix[:,0]]
    df1['Pub_Month'] = np.array([(x.split('.')[2]).split()[1] for x in df.ix[:,0]]
    df1['Pub_Day'] = np.array([(x.split('.')[2]).split()[2] for x in df.ix[:,0]]
    fifth_column = []
    for i in [f2(x) for x in df.ix[:,0]]:
        if len(i.split(';'))>1:
            fifth_column.append((i.split(';')[1]))
        else:
            fifth_column.append(' ')
    df1['Pub_Issue'] = np.array(fifth_column)
    df1['Title'] = df[1]
    df1['Author'] = df[2]
    df1['First_Author'] = df[2].apply(lambda x: x.split(',')[0])
    df1['Last_Author'] = df[2].apply(lambda x: x.split(',')[1])
    df1['Affiliation'] = df[3]
    df1['Abstract'] = df[4]
    journal_if = []
    for j_name in df1['Journal']:
        try:
            if_value = jif.ix[jif['Journal Name']==j_name, 'Impact Factor'][1]
        except IndexError:
            if_value = 1.0
        journal_if.append(if_value)
    df1['Impact_Factor'] = pd.Series(journal_if)
    df1.to_csv('pubmed_cleaned.txt', index=False, sep='|')


if __name__ == '__main__':
    file = open('pubmed_result.txt', 'r')
    text_cont_raw = file.read()
    abstract_list = initial_cleanup(text_cont_raw)
    adding_to_txt(abstract_list)
    cleaning_individual_fields()
    df1.to_csv('pubmed_cleaned.txt', index=False, sep='|')
