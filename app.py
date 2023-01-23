
import math
import os
from flask import Flask, render_template, request,send_file
from getdata import get_data
from collections import defaultdict
import ast
import tree
import variable_classifier
import python_ast_utils
import ast2vec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import traceback
matplotlib.use('Agg')


app = Flask(__name__)

programs     = ['']
traces       = []
model = None
perim=[]
X = None
tree_index = None
programs_to_trees = None
vc = None
trees = None
totno = 0
roll_list=None

def load_n_encode():
    global model, trees
    X = ast2vec.encode_trees(model, trees)
    return X

def check_data(rollno):
    if not os.path.exists(f'./mock_dataset/{rollno}'):
        os.mkdir(f'./mock_dataset/{rollno}')
    for f in os.listdir(f'./mock_dataset/{rollno}'):
        os.remove(os.path.join(f'./mock_dataset/{rollno}', f))

def get_trees():
    global programs, traces, programs_to_trees
    traces=[]
    programs=[]
    programs_to_trees=None

    student_dirs = list(sorted(os.listdir('mock_dataset')))
    totno = len(student_dirs)
    for student_dir in student_dirs:
        trace = [0]
        steps = list(sorted(os.listdir(f'mock_dataset/{student_dir}')))
        for step in steps:
            with open(f'mock_dataset/{student_dir}/{step}') as program_file:
                trace.append(len(programs))
                programs.append(program_file.read())
        traces.append(trace)
    print('read %d traces with %d programs' % (len(traces), len(programs)))
    trees, programs_to_trees = python_ast_utils.parse_asts(programs, filter_uniques = True)
    return trees

def print_ast(filename):
    with open(filename) as f:
        return (ast.dump(ast.parse(f.read())))

@app.route('/')
def home():
    global programs,traces,perim,X,tree_index,programs_to_trees,vc,trees,totno
    programs     = ['']
    traces       = []
    
    perim=[]
    X = None
    tree_index = None
    programs_to_trees = None
    vc = None
    trees = None
    totno = 0
    return render_template('index.html')

@app.route('/download')
def downloadFile ():
    path = os.getcwd()+"/student_ide/app.exe"
    return send_file(path, as_attachment=True)
    
@app.route('/student', methods=['POST', 'GET'])
def student_page():
    try:
        if request.method == "POST":
            if len(request.files.getlist('files[]'))!=0:
                files = request.files.getlist('files[]')
                for file in files:
                    filename = file.filename
                    rollno, stepno, prgno = filename.split('-')
                    check_data(rollno)
                for file in files:
                    filename = file.filename
                    rollno, stepno, prgno = filename.split('-')
                    new_name = f"./mock_dataset/{rollno}/"+f"step_{stepno}.txt"
                    file.save(new_name)

            else:
                files = dict(request.files)
                # file_name = rollno-traceno-prgno.txt
                for _,file in files.items():
                    filename = file.filename
                    rollno, stepno, prgno = filename.split('-')
                    check_data(rollno)
                for _,file in files.items():
                    filename = file.filename
                    rollno, stepno, prgno = filename.split('-')
                    if not os.path.exists(f'./mock_dataset/{rollno}'):
                        os.mkdir(f'./mock_dataset/{rollno}')
                    new_name = f"./mock_dataset/{rollno}/"+f"step_{stepno}.txt"
                    file.save(new_name)
            for root, dirs, files in os.walk('mock_dataset'):
                roll_list = dirs
                break
            roll_list.sort()   
            return '<h1> Successful Upload </h1>'
        else:
            return  render_template('student_page.html')

    except Exception as e:
        return render_template("err.html", error = e)  
		
@app.route('/educator')
def educator_page():
    global X, trees,tree_index,traces
    trees = get_trees() 
    tree_index = programs_to_trees[traces[0][-1]]
    tree_traces = [[0] + [programs_to_trees[program] for program in trace] for trace in traces]
    X = load_n_encode()
    cal_perim(tree_traces)
    create_marks()
    print("Length:",len(perim))
    return render_template('educator_page.html')

@app.route('/dashboard')
def dashboard_details():
    student_dirs = list(sorted(os.listdir('mock_dataset')))
    totno = len(student_dirs)
    
    return render_template('educator_page_old.html', tot_no = totno,get_data=get_data,marks=get_marks())

def get_marks():
    
    def default_value():
        return -2
    marks_dict=defaultdict(default_value)

    with open("marks.txt","r") as file:
        marks=file.readlines()
        for i in range(len(marks)):
            marks_dict[roll_list[i]]=marks[i].strip()
    print(marks_dict)
    return marks_dict

def create_marks():
    global perim
    marks=[]
    print(perim)
    p_copy=list(set(perim))
    p_copy.remove(max(p_copy))
    range=max(p_copy)-min(p_copy)
    mean = np.mean(p_copy)
    std = np.std(p_copy)
    if range==0:
        range=1
    for p in perim:
        if p == float('inf'):
            marks.append(-1)
        else:
            # marks.append(max(-30, (((max(p_copy)-p)/range)*100)))
            marks.append(min(100, math.floor((1-((p-mean)/std))*100)))
    with open("marks.txt","w") as file:
        file.writelines("%s\n" % data for data in marks)
    print(marks)
        
def cal_perim(traces):
    global X,tree_index,perim
    start=0
    perim=[]
    target=tree_index
    x = X[start, :]
    y = X[target, :]
    W, scale, _ = ast2vec.progress_pca(x, y, X)
    Xlo = np.dot(X - x, W.T) / scale
    
    start_included = np.all([trace[0] == start for trace in traces])
    end_included=[trace[-1]==target for trace in traces]
    print(end_included)
    for k in range(len(traces)):
        in_perim=0
        trace = traces[k]
        if not start_included:
            trace = [start] + trace
        if not end_included[k]:
            perim.append(float('inf')) 
            continue
        for t in range(1, len(trace)):

            i = trace[t-1]
            j = trace[t]
            
            diff=Xlo[j, :] - Xlo[i, :]
            in_perim+=math.sqrt(diff[0]**2+diff[1]**2)
        
        perim.append(in_perim)
    
    perim=[p-perim[0] for p in perim]
    print(perim)
# @app.route('/performance_details')
def handle_educator_request():
    
    global vc, model, trees
    # get the index of the program, which is the last program in the first trace
    #program_index = traces[0][-1]
    # get the corresponding syntax tree index
    

    vc = variable_classifier.VariableClassifier(model)
    vc.fit(trees);
    # try:
    #     plt.figure(figsize = (15, 10))
    #     ast2vec.interpolation_plot(model, start_tree = 0, target_tree = tree_index, X = X, variable_classifier = vc)
    #     plt.savefig('./static/images/intr_plt.png')
    #     plt.clf()

    #     tree_traces = [[0] + [programs_to_trees[program] for program in trace] for trace in traces]
    #     ast2vec.traces_plot(start = 0, target = tree_index, traces = tree_traces, X = X, trees = trees, plot_code = 3)
    #     plt.savefig('./static/images/trace_plt.png')
    #     plt.clf()

    #     W = ast2vec.construct_dynamical_system(tree_index, X, tree_traces)
    #     ast2vec.dynamical_system_plot(W, start_tree = 0, target_tree = tree_index, X = X, arrow_scale = 2., model = model, variable_classifier = vc)

    #     plt.savefig('./static/images/dynamic_plt.png')
    #     plt.clf()
    #     # return render_template('performance_details.html', graphJSON=graphJSON)
    #     return render_template('untitled1.html', name = 'new_plot', url ='/static/images/test.png')
    # except Exception as e:
    #     print('Error Occured')
    #     return str(traceback.print_exc())
    
@app.route('/interpolation_plot')
def interpolation_plot():
    global traces, tree_index, programs_to_trees, vc, model, trees, X
    if vc is None: 
        handle_educator_request()
    plt.figure(figsize = (15, 10))
    ast2vec.interpolation_plot(model, start_tree = 0, target_tree = tree_index, X = X, variable_classifier = vc)
    plt.savefig('./static/images/intr_plt.png')
    plt.clf()
    return render_template('intr.html')
    
@app.route('/trace_plot')
def trace_plot():
    global traces, tree_index, programs_to_trees, vc, model, trees, X,roll_list
    if vc is None: 
        handle_educator_request()
    tree_traces = [[0] + [programs_to_trees[program] for program in trace] for trace in traces]
    ast2vec.traces_plot(start = 0, target = tree_index, traces = tree_traces, X = X, roll_list=roll_list,trees = trees, plot_code = 10)
    plt.savefig('./static/images/trace_plt.png')
    plt.clf()
    return render_template('trace.html')


@app.route('/dynamic_plot')
def dynamic_plot():
    global traces, tree_index, programs_to_trees, vc, model, trees, X
    if vc is None: 
        handle_educator_request()
    tree_traces = [[0] + [programs_to_trees[program] for program in trace] for trace in traces]
    W = ast2vec.construct_dynamical_system(tree_index, X, tree_traces)
    ast2vec.dynamical_system_plot(W, start_tree = 0, target_tree = tree_index, X = X, arrow_scale = 2., model = model, variable_classifier = vc)
    plt.savefig('./static/images/dynamic_plt.png')
    plt.clf()
    return render_template('dynamic.html')

@app.route('/clus_plot')
def clus_plot():
    global traces, tree_index, programs_to_trees, vc, model, trees, X
    if vc is None: 
        handle_educator_request()

    dimred = PCA(n_components = 5)
    Xlo = dimred.fit_transform(X)
    print('our PCA model retains %g percent of the variance' % (100 * np.sum(dimred.explained_variance_ratio_)))

    n_clusters = 4
    clust = GaussianMixture(n_components = n_clusters, covariance_type = 'diag')
    Y = clust.fit_predict(Xlo)
    # extract the cluster means and map them back to the high-dimensional space
    means = dimred.inverse_transform(clust.means_)
    plt.figure(figsize = (15, 10))
    ast2vec.cluster_plot(start = 0, target = tree_index, X = X, Y = Y, means = means, model = model, variable_classifier = vc)
    plt.savefig('./static/images/clus_plt.png')
    plt.clf()
    return render_template('clust.html')


if __name__ == '__main__':
    for root, dirs, files in os.walk('mock_dataset'):
        roll_list = dirs
        break
    roll_list.sort()
    model = ast2vec.load_model()
    print('Model is loaded!')
    app.run(debug = True, port=5001)