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
    
    list_pid = [
        'LogadottirAmmonia2003','HonkalaAmmonia2005','CatappTrends2008','JiangTrends2009', 
        'WangUniversal2011','GrabowDescriptor-Based2011','StudtCO2012','BehrensThe2012',
        'FerrinHydrogen2012','MedfordElementary2012','MontoyaInsights2013','MedfordAssessing2014',
        'TangNickel--silver2014','FalsigOn2014','TsaiTuning2014','TsaiActive2014',
        'YooTheoretical2014','MedfordThermochemistry2014','ChanMolybdenum2014','Unpublished',
        'MontoyaThe2015','SeitzA2016','HoffmannFramework2016','YangIntrinsic2016',
        'GauthierSolvation2017','mgfieldslanders2018','MichalLixCoO22017','RolingConfigurational2017',
        'RolingBimetallic2017','FesterEdge2017','BukasORR2017','BoesAdsorption2018',
        'BajdichWO32018','SchumannSelectivity2018','BackPrediction2018','DickensElectronic2018','SandbergStrongly2018',
        'ChenUnderstanding2018','BackSingle2018','PatelTheoretical2018','HansenFirst2018',
        'ClarkInfluence2018','SniderRevealing2018','Park2D2019','ZhaoTrends2019','Schlexer2019pH',
        'SkafteSelective2019','SharadaAdsorption2019','GauthierImplications2019','StricklerSystematic2019',
        'JuUnraveling2019','ZhaoImproved2019','GauthierFacile2019','MeffordInterpreting2019',
        'TangFrom2020','FloresActive2020','HubertAcidic2020','TangModeling2020','SanchezCatalyst2020',
        'PengRole2020','GrewalHighly2019','LeeEpitaxial2020','BaeumerTuning2020','ZhengOrigin2020',
        'LandersUnderstanding2020','WangTheory-aided2020','PatelGeneralizable2021','JiangModelling2021',
        'WangAchieving2021','Gauthierrole2021','TangExploring2021','CamposEfficient2021','HalldinAssessing2021',
        'CraigHigh-throughput2021','Jia-ChengAtomistic2021','StreibelMicrokinetic2021',
        'ShiLong-term2021','PengTrends2022','LiuCatalytic2022','LiScreening2021','AraComputational2022',
        'ComerUnraveling2022','LungerCation-dependent2022','SainiElectronic2022','TettehCompressively2022',
        'KaiData-driven2022','HossainInvestigation2022','KoshyInvestigation2022','WeiInsights2022',
        'RaoResolving2022',
        # 'MamunHighT2019',    
    ]
    curdir = '/curdir/' ## ''
    for ipid, pid in enumerate(list_pid):
        print(ipid, pid, len(list_pid))
        raw_reactions = reactions_from_dataset(pid) 
        reactions = copy.deepcopy(raw_reactions)
        aseify_reactions(reactions)

        # Save reactions to a file
        with open(f'{curdir}v2_script_10/exp12/reactions_{pid}.pickle', 'wb') as f:
            pickle.dump(reactions, f)

        # # Load reactions from a file
        # with open('/curdir/datasets/reactions.pickle', 'rb') as f:
        #     loaded_reactions = pickle.load(f)
    