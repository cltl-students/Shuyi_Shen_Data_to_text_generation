import json


"""
{'text': 'Advanced skills in HTML5 and SASS / SCSS to W3C standards \n Advanced skills in OO JavaScript and JavaScript libraries such as jQuery \n Advanced skills of Ajax / JSON and REST APIs \n Experience in Responsive Development \n Experience in Agile Environment \n Experience of Photoshop and Adobe XD to be able to manipulate designs \n Experience using JSPs and JSTL and XML \n Experience with Jira \n Experience with source control software such as TortoiseHg / Mercurial \n Experience with JS Frameworks ( VueJS , ReactJs or similar ) \n Experience with Grunt / Gulp task runner \n Basic experience with Linux commands', 
'triple_list': [['Experience', 'Experience_skills', 'Responsive Development'], ['Experience', 'Experience_skills', 'Agile Environment'], ['Experience', 'Experience_skills', 'Photoshop'], ['Experience', 'Experience_skills', 'Adobe XD'], ['be able', 'knowledge_skills', 'manipulate designs'], ['Experience', 'Experience_skills', 'using JSPs and JSTL and XML'], ['Experience', 'Experience_skills', 'Jira'], ['Experience', 'Experience_skills', 'source control software'], ['Experience', 'Experience_skills', 'TortoiseHg / Mercurial'], ['Experience', 'Experience_areas', 'JS Frameworks ( VueJS , ReactJs or similar )'], ['Experience', 'Experience_skills', 'Grunt / Gulp task runner'], ['Basic experience', 'Experience_skills', 'Linux commands']]}


"""
source_paths = ["train_triples.json","val_triples.json","test_triples.json","temp_triples.json"]
# data = load_json(source_path)
for source_path in source_paths:
    list_final = []
    with open(source_path ,'r', encoding='utf-8') as f:
        data = json.load(f)
    for i in data:
        dict_sen = {}
        sentenc_list = i['text'].split('\n')
        triple_list = i['triple_list']
        for triple in triple_list:
            for idx,sentenc in enumerate(sentenc_list):
                if triple[2] in sentenc:
                    dict_tmp = dict_sen.get(str(idx),[])
                    dict_tmp.append(triple)
                    dict_sen[str(idx)] = dict_tmp
                    break
        for s_idx,val in dict_sen.items():
            dict_tmp ={
                'text': sentenc_list[int(s_idx)],
                'triple_list': val
            }
            list_final.append(dict_tmp)
    # name = source_path.split('.')[0] + "_per_line." + source_path.split('.')[1]
    print('dump data:{}'.format(source_path))
    with open(source_path ,'w',encoding='utf-8') as f:
        json.dump(list_final, f, indent=4)




