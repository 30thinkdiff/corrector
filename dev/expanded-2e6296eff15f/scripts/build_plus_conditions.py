"""CPU selection from fixed official metadata, never from rollout outcomes."""
import argparse,ast,json,re,subprocess
from pathlib import Path

SHA='4976dc30028e805ff8094b55501d532c48fec182'
BASES=['pick_up_the_alphabet_soup_and_place_it_in_the_basket','pick_up_the_cream_cheese_and_place_it_in_the_basket']

def select(source):
    root=source/'libero/libero';folder=root/'benchmark'
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SHA:raise ValueError('fixed Plus source required')
    tree=ast.parse((folder/'libero_suite_task_map.py').read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets))
    names=ast.literal_eval(assignment.value)['libero_object']
    lookup={name:i for i,name in enumerate(names)}
    rows=json.loads((folder/'task_classification.json').read_text())['libero_object'];conditions={}
    def add(tag,name,category,level=None,historical=False,default=False):
        filename=name.split('_view_',1)[0]+'.bddl'
        path=root/'bddl_files/libero_object'/filename
        match=re.search(r'\(:language\s+(.*?)\)',path.read_text(),re.S)
        if not match:raise ValueError('actual BDDL instruction missing')
        language=' '.join(match[1].split()).strip('"').lower()
        row={'task_id':lookup[name],'task_name':name,'init':0,'language':language,'category':category,
             'official_difficulty':level,'historical_compat':historical}
        if default:row['default_task']=BASES[0]
        conditions['plus/'+tag]=row
    for tag,index in [('default',653),('view15',666),('view5',656),('view10',661)]:
        add(tag,names[index],'Historical camera',historical=True,default=tag=='default')
    for base in BASES:
        grouped={}
        for row in rows:
            if row['name'].startswith(base+'_'):grouped.setdefault(row['category'],[]).append(row)
        if len(grouped)!=7:raise ValueError('base lacks one perturbation category')
        for category,items in sorted(grouped.items()):
            levels=sorted({r['difficulty_level'] for r in items});selected=[]
            for level in [levels[0],levels[-1]]:
                candidates=sorted((r for r in items if r['difficulty_level']==level),key=lambda r:r['name'])
                row=next((r for r in candidates if r['name'] not in {s['name'] for s in selected}),None)
                if row: selected.append(row)
            if len(selected)<2:
                row=next((r for r in sorted(items,key=lambda r:r['name']) if r['name'] not in {s['name'] for s in selected}),None)
                if row:selected.append(row)
            for n,row in enumerate(selected):
                add(('soup' if base==BASES[0] else 'cheese')+'/'+category.lower().replace(' ','_')+'/'+str(n+1),row['name'],category,row['difficulty_level'])
    return {'source_sha':SHA,'selection':'two predeclared base tasks, official min/max available difficulty, lexical first name; no success data',
            'task_index_note':'classification id is one-based; actual suite ID resolved by exact name, never copied blindly',
            'methods':17,'seeds':[1000,1001,1002],'conditions':conditions,'planned':len(conditions)*3*17,
            'scope':'structured diagnostic subset, not official 10030-task leaderboard protocol'}

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=select(a.source);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(len(result['conditions']),result['planned'])
