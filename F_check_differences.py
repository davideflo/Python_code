# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:10:57 2017

@author: d_floriello

check differences Terna/PDOs
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
import datetime


###############################################################################
def GetData(pdos):
    diz = OrderedDict()
    for i in range(pdos.shape[0]):
        ll = []
        pod = pdos['Pod'].ix[i]
        dt = datetime.date(int(pdos['MeseAnno'].ix[i][3:]), int(pdos['MeseAnno'].ix[i][:2]), int(pdos['Ea'].ix[i]))
        zona = pdos['PuntoDispacciamento'].ix[i]
        ll.append(pod)
        ll.append(dt)  
        ll.append(zona)
        vec = np.repeat(0.0, 24)
        for h in range(24):
            vec[h] = float(pdos[str(h) + '.A'].ix[i].replace(',','.')) + float(pdos[str(h) + '.B'].ix[i].replace(',','.')) + float(pdos[str(h) + '.C'].ix[i].replace(',','.')) + float(pdos[str(h) + '.D'].ix[i].replace(',','.')) 
        ll.extend(vec.tolist())
        diz[i] = ll
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['Pod', 'DATA','zona','1','2','3','4','5','6','7','8','9','10','11','12','13',
                    '14','15','16','17','18','19',
                    '20','21','22','23','24']]
    return diz
###############################################################################
def ToTS(Pdo,zona):
    Pdoz = Pdo.ix[Pdo['zona'] == zona]
    dr = pd.date_range('2017-01-01', '2017-04-01', freq = 'H')[:2160]
    diz = OrderedDict()
    for d in dr:
        dd = d.date()
        sum_at_day = Pdoz.ix[Pdoz['DATA'] == dd].sum()
        h = d.hour
        diz[d] = sum_at_day.ix[h]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    return diz
###############################################################################
def convertDates(vec):
   CD = vec.apply(lambda x: datetime.datetime(year = int(str(x)[6:10]), month = int(str(x)[3:5]), day = int(str(x)[:2]), hour = int(str(x)[11:13])))
   return CD
###############################################################################


terna = pd.read_excel("C:/users/d_floriello/Documents/aggregato_sbilanciamento2.xlsx")
pdos = pd.read_excel("C:/users/d_floriello/Desktop/XML_Importer.xlsm")

pdos.to_hdf("C:/users/d_floriello/Desktop/pdos.h5", "PDO")

cl = [x for x in pdos.columns[:8].tolist()]
for h in range(24):
    cl.append(str(h) + '.A')
    cl.append(str(h) + '.B')
    cl.append(str(h) + '.C')
    cl.append(str(h) + '.D')
pdos.columns = cl

pdos = pdos.fillna(value = '0,0')
Pdo = GetData(pdos)

Pdo = Pdo.ix[Pdo['DATA'] > datetime.date(2016,12,31)]
Pdo = Pdo.ix[Pdo['CodFlusso'] == 'PDO']

nord = ToTS(Pdo, "NORD")
nord = nord.apply(lambda x: x/1000)
nord.plot()


tnord = terna.ix[terna['CODICE RUC'] == 'UC_DP1608_NORD']
tnord.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:tnord.shape[0]]
cd = convertDates(tnord['DATA RIFERIMENTO CORRISPETTIVO'])
tnord = tnord.set_index(cd.values)

tnord['MO [MWh]'].plot()

tnord = tnord.ix[tnord.index.date > datetime.date(2016,12,31)]
tnord = tnord.ix[tnord.index.date < datetime.date(2017,4,1)]

diz = OrderedDict()
dr = pd.date_range('2017-01-01', '2017-04-01', freq = 'H')[:2160]
for i in dr:
    if i in tnord.index:
        if tnord.ix[i].shape[0] > 1:
            y = tnord['MO [MWh]'].ix[i].sum()
        elif tnord.ix[i].shape[0] == 0:
            y = 0
        else:
            y = tnord['MO [MWh]'].ix[i]
    else:
        y = 0
    diz[i] = [y]

Tnord = pd.DataFrame.from_dict(diz, orient = 'index')

Tnord.plot(title = 'NORD - TERNA')

diff = nord - Tnord
diff.plot(color = 'green', title = 'differenza PDO-Terna')

diff.mean()
diff.std()
diff.median()

pd.tools.plotting.autocorrelation_plot(diff)


import xml.etree.ElementTree as ET
import os
import glob
path2 = 'Z:/AREA BO&B/23.P-RNO - PD-RFO DISTRIBUTORI - BONUS/2017/01. Invio di Gennaio/SOS/'

for infile in glob.glob( os.path.join(path, '*.xml') ):
        tree = ET.parse(infile)
        root = tree.getroot()
        with open(infile+'new.csv','w') as outfile:
            for elem in root.findall('.//event[@type="MEDIA"]'):
                    mediaidelem = elem.find('./mediaid')
                    if mediaidelem is not None:
                            outfile.write("{}\n".format(mediaidelem.text))

for infile in glob.glob( os.path.join(path, '*.xml') ):
    try:
        tree = ET.parse(infile)
        root = tree.getroot()
        print(root)
    except ET.ParseError as e:
        print(infile, str(e))
        continue
###############################################################################
import os
import xml.parsers.expat
from xml.sax.saxutils import escape
from optparse import OptionParser
from math import log10


# How much data we process at a time
CHUNK_SIZE = 100 * 1024 * 1024

# The sequence of element leading us to the current one
path = []

# How far we are in the current file
cur_size = 0
# From how much should we start another file
MAX_SIZE = 100 * 1024*1024 # 1Mb

# The current index
cur_idx = 0
# The current file handle we are writing to
cur_file = None

# The format string used to introduce the index in the file to be written
FMT = ".%d"

# The filename we are playing with
out_dir = None
root = None
ext = None

# The xml declaration of the file.
xml_declaration = None

# What was the signature of the last start element
start = None

# if we are currently in the process of changing file
ending = False

def attrs_s(attrs):
    """ This generate the XML attributes from an element attribute list """
    l = ['']
    for i in range(0,len(attrs), 2):
        l.append('%s="%s"' % (attrs[i], escape(attrs[i+1])))
    return ' '.join(l)

def next_file():
    """ This makes the decision to cut the current file and starta new one """
    global cur_size, ending
    if (not ending) and (cur_size > MAX_SIZE):
        # size above threshold, and not already ending
        global cur_file, cur_idx
        print("part %d Done" % cur_idx)
        ending = True
        # Close the current elements
        for elem in reversed(path):
            end_element(elem[0])
        # Close the file
        cur_file.close()
        # reset the size
        cur_size = 0
        # Open another file
        cur_idx += 1
        cur_file = open(os.path.join(out_dir, root + FMT % cur_idx + ext),
                        'wt')
        if xml_declaration is not None:
            cur_file.write('<?xml%s?>\n' % attrs_s(xml_declaration))
        # Start again where we stopped
        for elem in path:
            start_element(*elem)
        # We are done 'ending'
        ending = False


def xml_decl(version, encoding, standalone):
    global xml_declaration
    l = ['version', version, 'encoding', encoding]
    if standalone != -1:
        l.extend(['standalone', 'yes' if standalone else 'no'])
    xml_declaration = l
    cur_file.write('<?xml%s?>\n' % attrs_s(xml_declaration))


def start_element(name, attrs):
    """ Called by the parser when he meet a start element """
    global cur_size, start
    if start is not None:
        # Chaining starts after each others
        cur_file.write('<%s%s>' % (start[0], attrs_s(start[1])))
    start = (name, attrs)
    if ending:
        return
    cur_size += len(name) + sum(len(k) for k in attrs)
    path.append((name, attrs))


def end_element(name):
    """ Caled by the parser when he meet an end element """
    global cur_size
    global start
    if start is not None:
        # Empty element, good, we did not wrote the start part
        cur_file.write('<%s%s/>' % (start[0],attrs_s(start[1])))
    else:
        # There was some data, close it normaly
        cur_file.write('</%s>' % name)
    start = None
    if ending:
        return
    elem = path.pop()
    assert elem[0] == name
    cur_size += len(name)
    next_file()


def char_data(data):
    """ Called by the parser when he meet data """
    global cur_size, start
    wroteStart = False
    if start is not None:
        # The data belong to an element, we should write the start part first
        cur_file.write('<%s%s>' % (start[0], attrs_s(start[1])))
        start = None
        wroteStart = True
    # ``escape`` is too much for us, only & and < ned to be escaped there ...
    data = data.replace('&', '&amp;')
    data = data.replace('<', '&lt;')
    if data == '>':
        data = '&gt;'
    cur_file.write(data.encode('utf-8'))
    cur_size += len(data)
    if not wroteStart:
        # The data was outside of an element, it could be the right moment to
        # make the split
        next_file()

def main(filename, output_dir):
    # Create a parser
    p = xml.parsers.expat.ParserCreate()
    # We want to reproduce the input, so we are interested in the order of the
    # attributess
    p.ordered_attributes = 1

    # Set our callbacks (we are stripping comments out by not defining
    # callbacks for them)
    p.XmlDeclHandler = xml_decl
    p.StartElementHandler = start_element
    p.EndElementHandler = end_element
    p.CharacterDataHandler = char_data

    global cur_file, cur_idx
    global out_dir, root, ext

    global FMT
    FMT = ".%%0%dd" % (int(log10(os.path.getsize(filename) / MAX_SIZE)) + 1)
    
    filename2 = filename
    out_dir, filename = os.path.split(filename)
    if output_dir is not None:
        out_dir = output_dir

    root, ext = os.path.splitext(filename)

    cur_file = open(os.path.join(out_dir, root + FMT % cur_idx + ext), 'wt')

    with open(filename2, 'rt') as xml_file:
        while True:
            # Read a chunk
            chunk = xml_file.read(CHUNK_SIZE)
            if len(chunk) < CHUNK_SIZE:
                # End of file
                # tell the parser we're done
                p.Parse(chunk, 1)
                # exit the loop
                break
            # process the chunk
            p.Parse(chunk)
            
    # Don't forget to close our handle
    cur_file.close()
    
    print("part %d Done" % cur_idx        )
    
    
if __name__ == "__main__":
    parser = OptionParser(usage="usage: %prog [options] XML_FILE")
    parser.add_option("-o", "--output-dir",
        help="Specify the directory where the xml files will be written" \
            "(default to the same directory where the original file is)")
    parser.add_option("-M", "--max_size", type="int",
        help="Specify the size at which the files should be split (in Kb)")
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error("incorrect number of arguments")
    if options.max_size is not None:
        MAX_SIZE = options.max_size * 1024
    main(args[0], options.output_dir)
        
    main(infile, path2)
        