import argparse
import os
import pandas as pd


def readdata(filepath, filename):
    
    """
    Reads and processes records from the specified file.
    
    Parameters:
    filepath (str): Directory path to the file.
    filename (str): Name of the file to read.
    
    Returns:
    list: A list of processed records, each represented as a list of attributes.
    
    Exceptions:
    Prints an error message if the file cannot be opened or attributes cannot be converted to integers.
    """
    
    records = []
    try:
        with open(os.path.join(filepath, filename), 'r') as rf:
            next(rf)
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                line = [a.strip() for a in line.split(',')]
                intidx = [ATTNAME.index(colname) for colname in (
                    'age', 'fnlwgt', 'education_num', 'capital-gain', 'capital-loss', 'hours-per-week')]
                for idx in intidx:
                    try:
                        line[idx] = int(line[idx])
                    except:
                        print('attribute %s, value %s, cannot be converted to number' %(ATTNAME[idx], line[idx]))
                        line[idx] = -1
                for idx in range(len(line)):
                    if line[idx] == '' or line[idx] == '?':
                        line[idx] = '*'
                records.append(line)
        return records
    except:
        print('cannot open file: %s:%s' %(filepath, filename))
    

def is_k_anonymous(df, quasi_identifiers, k):
    
    """
    Determines if a DataFrame is k-anonymous based on given quasi-identifiers.
    
    Parameters:
    df (DataFrame): The DataFrame to be checked.
    quasi_identifiers (list): A list of column names that are considered quasi-identifiers.
    k (int): The desired level of anonymity.
    
    Returns:
    bool: Returns True if the DataFrame is k-anonymous based on the quasi-identifiers and k, otherwise False.
    """
    
    groups = df.groupby(quasi_identifiers).size().reset_index(name='counts')
    
    if groups['counts'].min() < k:
        return False
    
    return True



class KAnonymity():
    def __init__(self, records):
        self.records = records
        self.confile = [AGECONFFILE, EDUCONFFILE, MARITALCONFFILE, RACECONFFILE]
        
    def anonymize(self, qi_names=['age', 'education', 'marital-status', 'race'], k=5):
        """
        anonymizer for k-anonymity
        
        Keyword Arguments:
            qi_names {list} -- [qi names] (default: {['age', 'education', 'marital-status', 'race']})
            k {int} -- [value for k] (default: {5})
        """

        domains, gen_levels = {}, {}
        qi_frequency = {}       # store the frequency for each qi value
        # record_att_gen_levels = [[0 for _ in range(len(qi_names))] for _ in range(len(self.records))]

        assert len(self.confile) == len(qi_names), 'number of config files  not equal to number of QI-names'
        generalize_tree = dict()
        for idx, name in enumerate(qi_names):
            generalize_tree[name] = Tree(self.confile[idx])

        for qiname in qi_names:
            domains[qiname] = set()
            gen_levels[qiname] = 0
        
        for idx, record in enumerate(self.records):
            qi_sequence = self._get_qi_values(record[:], qi_names, generalize_tree)
            
            if qi_sequence in qi_frequency:
                qi_frequency[qi_sequence].add(idx)
            else:
                qi_frequency[qi_sequence] = {idx}
                for j, value in enumerate(qi_sequence):
                    domains[qi_names[j]].add(value)
        
        # iteratively generalize the attributes with maximum distinct values
        while True:
            # count number of records not satisfying k-anonymity
            negcount = 0
            for qi_sequence, idxset in qi_frequency.items():
                if len(idxset) < k:
                    negcount += len(idxset)
            
            if negcount > k:
                # continue generalization, since there are more than k records not satisfying k-anonymity
                most_freq_att_num, most_freq_att_name = -1, None
                for qiname in qi_names:
                    if len(domains[qiname]) > most_freq_att_num:
                        most_freq_att_num = len(domains[qiname])
                        most_freq_att_name = qiname
                
                # find the attribute with most distinct values
                generalize_att = most_freq_att_name
                qi_index = qi_names.index(generalize_att)
                domains[generalize_att] = set()
                
                # generalize that attribute to one higher level
                for qi_sequence in list(qi_frequency.keys()):
                    new_qi_sequence = list(qi_sequence)
                    new_qi_sequence[qi_index] =  generalize_tree[generalize_att].root[qi_sequence[qi_index]][0]
                    new_qi_sequence = tuple(new_qi_sequence)
                
                    if new_qi_sequence in qi_frequency:
                        qi_frequency[new_qi_sequence].update(
                            qi_frequency[qi_sequence])
                        qi_frequency.pop(qi_sequence, 0)
                    else:
                        qi_frequency[new_qi_sequence] = qi_frequency.pop(qi_sequence)
                    
                    domains[generalize_att].add(new_qi_sequence[qi_index])
                
                gen_levels[generalize_att] += 1
                
            
            else:
                # end the while loop
                # suppress sequences not satisfying k-anonymity
                # save results and calculate distoration and precision
                genlvl_att = [0 for _ in range(len(qi_names))]
                dgh_att = [generalize_tree[name].level for name in qi_names]
                datasize = 0
                qiindex = [ATTNAME.index(name) for name in qi_names]

                # used to make sure the output file keeps the same order with original data file
                towriterecords = [None for _ in range(len(self.records))]
                with open(f"../../data/{filename.replace('.csv', '')}_{k}_anonymized.csv", 'w') as wf:
                    column_names = ATTNAME
                    wf.write(', '.join(column_names))
                    wf.write('\n')
                    for qi_sequence, recordidxs in qi_frequency.items():
                        if len(recordidxs) < k:
                            continue
                        for idx in recordidxs:
                            record = self.records[idx][:]
                            for i in range(len(qiindex)):
                                record[qiindex[i]] = qi_sequence[i]
                                genlvl_att[i] += generalize_tree[qi_names[i]].root[qi_sequence[i]][1]
                            record = list(map(str, record))
                            for i in range(len(record)):
                                if record[i] == '*' and i not in qiindex:
                                    record[i] = '?'
                            towriterecords[idx] = record[:]
                            # wf.write(', '.join(record))
                            # wf.write('\n')
                        datasize += len(recordidxs)
                    for record in towriterecords:
                        if record is not None:
                            wf.write(', '.join(record))
                            wf.write('\n')
                        else:
                            wf.write('\n')
                
                print('Quasi Identifiers: ', qi_names)
                # precision = self.calc_precission(genlvl_att, dgh_att, datasize, len(qi_names))
                precision = self.calc_precision(genlvl_att, dgh_att, len(self.records), len(qi_names))
                distoration = self.calc_distoration([gen_levels[qi_names[i]] for i in range(len(qi_names))], dgh_att, len(qi_names))
                print("Anonymization Completed!")
                print(f"Precision of anonymization: {precision:.3f}")
                print(f"Distoration of anonymization: {distoration:.3f}")
                print(f"Anonymized dataset is saved at: '../../data/{filename.replace('.csv', '')}_{k}_anonymized.csv'")

                break


    def calc_precision(self, genlvl_att, dgh_att, datasize, attsize = 4):
        """
        calculate the precision of generalized value for each value of each attributes
        
        Arguments:
            genlvl_att {[list]} -- [sum of generalized level of each attribute]
            dgh_att {[list]} -- [maximum height of each attribute]
            datasize {[int]} -- [data size]
        
        Keyword Arguments:
            attsize {int} -- [number of qi attributes] (default: {4})
        
        Returns:
            [float] -- [precision of the generalization]
        """

        return 1 - sum([genlvl_att[i] / dgh_att[i] for i in range(attsize)])/(datasize*attsize)


    def calc_distoration(self, gen_levels_att, dgh_att, attsize):
        """
        calculate the distoration for generalized levels of each attributes
        
        Arguments:
            gen_levels_att {[type]} -- [description]
            dgh_att {[type]} -- [description]
            attsize {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        # print('attribute gen level:', gen_levels_att)
        # print('tree height:', dgh_att)
        return sum([gen_levels_att[i] / dgh_att[i] for i in range(attsize)]) / attsize

        
    def _get_qi_values(self, record, qi_names, generalize_tree):
        """
        private method
        get qi values from one record
        
        Arguments:
            record {[list]} -- [one record]
            qi_names {[list]} -- [qi names]
            generalize_tree {[dict]} -- [dict storing the DGH trees]
        
        Returns:
            [tuple] -- [qi tuple value]
        """

        qi_index = [ATTNAME.index(name) for name in qi_names]
        seq = []
        for idx in qi_index:
            if idx == ATTNAME.index('age'):
                if record[idx] == -1:
                    seq.append('0-100')
                else:
                    seq.append(str(record[idx]))
            else:
                if record[idx] == '*':
                    # TODO, handle missing value cases
                    record[idx] = generalize_tree[qi_names[idx]].highestgen
                seq.append(record[idx])
        return tuple(seq)

            

        
class Tree:
    """
    Tree class
    built for DGH tree, keep track of each node's parent, and current level
    """

    def __init__(self, confile):
        self.confile = confile
        self.root = dict()
        self.level = -1
        self.highestgen = ''
        self.buildTree()
        
    
    def buildTree(self):
        """
        build the DGH tree from config file
        """

        with open(self.confile, 'r') as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                line = [col.strip() for col in line.split(',')]
                height = len(line)-1
                if self.level == -1:
                    self.level = height
                if not self.highestgen:
                    self.highestgen = line[-1]
                pre = None
                for idx, val in enumerate(line[::-1]):
                    self.root[val] = (pre, height-idx)
                    pre = val
                



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--filename', type=str, default='adults_syn_ctgan.csv')
    parser.add_argument('--filepath', type=str, default='../../data')
    args = parser.parse_args()
    k = args.K
    filename = args.filename
    filepath = args.filepath

    print(f"Executing k-Anonymization on the dataset using a k-value of {k}.")


    ATTNAME = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']

    AGECONFFILE = '../../conf/age_hierarchy.txt'
    EDUCONFFILE = '../../conf/edu_hierarchy.txt'
    MARITALCONFFILE = '../../conf/marital_hierarchy.txt'
    RACECONFFILE = '../../conf/race_hierarchy.txt'

    quasi_identifiers=['age', 'education', 'marital-status', 'race']

    
    data = readdata(filepath, filename)
        
    KAnony = KAnonymity(data)
    KAnony.anonymize(quasi_identifiers, k)
