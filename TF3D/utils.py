import pandas as pd

def get_method_path(idx_list, features):
    return [(features[idx]['path'], features[idx]['method'], features[idx]['beginline']) for idx in idx_list]

def generate_bat(l, name = "sublime.bat"):
    """Take list of paths and generate a batch file to directly open the specified paths"""
    myBat = open("sublime.bat",'w+')
    for element in l:
        myBat.write('start ' + DIR_ + element[3:] + '\n')
    myBat.close()

def get_script_name(path):
    p = list(reversed(path))
    result = []
    for el in p[3:]:
        if el == '\\':
            return ''.join(list(reversed(result)))
        else:
            result.append(el)

def get_project_name(path):
    try:
        p = path.split('\\')[2]
    except:
        p = path.split('/')[2]
    return p
        

def create_docstring_dataframe(features, list_dstr):
	list_scripts = []
	list_paths = []
	list_methods = []
	list_projects = []
	list_features = []
	list_sizecodes = []
	
	for el in features:
	    list_scripts.append(get_script_name(el['path']))
	    list_paths.append(el['path'])
	    list_methods.append(el['method'])
	    list_projects.append(get_project_name(el['path']))
	    list_features.append(el['features'])
	    list_sizecodes.append(el['endline'] - el['beginline']+1)

	
	data = pd.DataFrame({'index' : [i for i in range(len(features))], 'path' : list_paths, 'scripts' : list_scripts, 'methods' : list_methods, 'projects' : list_projects, 'features' : list_features, 'code_lines' : list_sizecodes, 'docstring' : list_dstr})
	
	return data
