# https://nbviewer.org/gist/mhoffman/c418acb6b3f928eb4ef71f4001d120d9

import requests
import pprint
import sys
import string
import json
import io
import copy
import ase.io
import ase.calculators.singlepoint
import pickle

GRAPHQL = 'http://api.catalysis-hub.org/graphql'

def fetch(query):
    return requests.get(
        GRAPHQL, {'query': query}
    ).json()['data']

def reactions_from_dataset(pub_id, page_size=10):
    reactions = []
    has_next_page = True
    start_cursor = ''
    page = 0
    while has_next_page:
        data = fetch("""{{
      reactions(pubId: "{pub_id}", first: {page_size}, after: "{start_cursor}") {{
        totalCount
        pageInfo {{
          hasNextPage
          hasPreviousPage
          startCursor
          endCursor 
        }}  
        edges {{
          node {{
            chemicalComposition
            surfaceComposition
            facet
            sites
            coverages
            reactants
            products
            Equation
            reactionEnergy
            activationEnergy
            dftCode
            dftFunctional
            username
            pubId
            reactionSystems {{
              name
              systems {{
                energy
                InputFile(format: "json")
              }}
            }}  
          }}  
        }}  
      }}    
    }}""".format(start_cursor=start_cursor,
                 page_size=page_size,
                 pub_id=pub_id,
                ))
        has_next_page = data['reactions']['pageInfo']['hasNextPage']
        start_cursor = data['reactions']['pageInfo']['endCursor']
        page += 1
        print(has_next_page, start_cursor, page_size * page, data['reactions']['totalCount'])
        reactions.extend(map(lambda x: x['node'], data['reactions']['edges']))

    return reactions

def aseify_reactions(reactions):
    for i, reaction in enumerate(reactions):
        for j, _ in enumerate(reactions[i]['reactionSystems']):
            with io.StringIO() as tmp_file:
                system = reactions[i]['reactionSystems'][j].pop('systems')
                tmp_file.write(system.pop('InputFile'))
                tmp_file.seek(0)
                atoms = ase.io.read(tmp_file, format='json')
            calculator = ase.calculators.singlepoint.SinglePointCalculator(
                atoms,
                energy=system.pop('energy')
            )
            atoms.set_calculator(calculator)
            #print(atoms.get_potential_energy())
            reactions[i]['reactionSystems'][j]['atoms'] = atoms
        # flatten list further into {name: atoms, ...} dictionary
        reactions[i]['reactionSystems'] = {x['name']: x['atoms']
                                          for x in reactions[i]['reactionSystems']}
        
if __name__=='__main__':
    raw_reactions = reactions_from_dataset("MamunHighT2019") # this will get all 88587 reactions
    # raw_reactions = reactions_from_dataset("FesterEdge2017")
    
    reactions = copy.deepcopy(raw_reactions)
    aseify_reactions(reactions)
    
    # Save reactions to a file
    with open('/curdir/datasets/reactions.pickle', 'wb') as f:
        pickle.dump(reactions, f)
    
    # # Load reactions from a file
    # with open('/curdir/datasets/reactions.pickle', 'rb') as f:
    #     loaded_reactions = pickle.load(f)
    