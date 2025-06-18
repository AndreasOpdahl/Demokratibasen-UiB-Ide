# Author: Lars Næsheim
# Date: 20240418

import subprocess
import re
import magic
import shutil
import io
import os
import tempfile

class PdfExtraction:
    ''' PDF converter
    class for converting pdf to text,xml,and also a rudimentary html;
    html is dependent on the documant having structural tags.
    The class methods are generally wrappers around linux command line
    utilities, and thus they must be installed. These utilities all depend
    on the poppler library, which is written in c++. The main reason for
    selecting poppler and for using the CLI utility wrappers is to be able
    to extract structural info. AFAIK only poppler by way of pdfinfo can
    do this.
    Extracting the structural info is relatively time-consuming.
    Extracting pure text and xml is much quicker.

    If run as standalone, a simple test is run, either on a file given
    as a command line arg, or else on a trivial PDF file generated with pandoc

    linux package dependencies (debian):
    * poppler-utils
    * pandoc (testing only)

    TODOS:
    * add sane error handling. Possibly add an 'error' field to the
    object to store error messages from the CLI utilities?
    '''
    def __init__(self, pdf_input):
        ''' Takes either a filename, a file handle, or a byte array as input. '''
        if type(pdf_input) == io.BufferedReader or type(pdf_input) == bytes:
            self._temp = tempfile.NamedTemporaryFile(delete=False)
            buf = pdf_input.read() if type(pdf_input) == io.BufferedReader else pdf_input
            self._temp.write(buf)
            self._temp.close()
            self.filename = self._temp.name
        else:
            self._temp = None
            self.filename = pdf_input
        self._is_pdf = magic.from_file(self.filename).startswith("PDF document")
        self.info = dict()
        self.img = None
        self.text = ''
        self.xml = ''
        self.html = ''
        utils = ('pdfinfo', 'pdftotext', 'pdftohtml', 'pdftoppm')
        missing = [util for util in utils if not shutil.which(util)]
        if missing:
            print("Must install missing "+", ".join(missing))

    def is_pdf(self):
        return self._is_pdf

    def cleanup(self):
        ''' Removes temp-pdf-file created when input is file handle or bytes '''
        if self._temp and os.path.isfile(self._temp.name): 
            os.unlink(self._temp.name)
    
    def run(self):
        ''' Convenience function to run all types of extractions
        from the PDF file.
        '''
        if not self._is_pdf:
            return self
        self.get_info()
        self.to_text()
        self.to_xml()
        self.to_html()
        self.to_thumb()
        return self

    def get_info(self):
        ''' compiles the different types of metadata '''
        self._get_img_info()
        self._get_meta_info()
        self._is_scanned()
        return self.info

    def to_text(self):
        ''' Wrapper around pdftotext -layout ''' 
        if not self._is_pdf:
            return None
        cmd=["pdftotext", "-layout", self.filename, "-"]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        self.text = ret.stdout
        return self.text

    def to_xml(self):
        ''' Wrapper around pdftotext -xml '''
        if not self._is_pdf:
            return None
        cmd=["pdftohtml", "-xml", "-stdout", self.filename]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        self.xml = ret.stdout    
        return self.xml
        
    def to_html(self):
        ''' Parses pdfinfo -struct-text into an html'''
        if not self._is_pdf:
            return ''
        if  not (self.info.get('meta') or self._get_meta_info().get('meta')).get('Tagged',False):
            return ''
        cmd = ['pdfinfo', '-struct-text', self.filename]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        cur_indent = 0
        root = Node(None,"root")
        cur_parent = root
        prev_node = cur_parent
        for line in ret.stdout.split("\n"):
            indent = (len(line) - len(line.lstrip()))//2
            if indent > cur_indent:
                cur_parent = prev_node
            if indent < cur_indent:
                for i in range(indent,cur_indent):
                    cur_parent = cur_parent.parent
            node = Node(cur_parent,line.strip())
            node.level = indent
            cur_indent=indent
            cur_parent.add_child(node)
            prev_node = node
        doc=root.children[0]
        ret = [ "<html><head></head>" ]
        ret = doc.build(ret)
        ret.append( "</html>")
        self.html = "\n".join(ret)
        return self.html

    def to_thumb(self):
        ''' Makes a small PNG of the first page of the PDF '''
        cmd = ["pdftoppm", "-png", "-singlefile", "-r", "50", self.filename]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        self.img = res.stdout
        return self.img

    def _get_img_info(self):
        ''' gets info about images in the file:
        * how many
        * total area of all images
        * if the first page of the PDF is an image '''
        cmd=["pdfimages", "-list", self.filename]
        res = subprocess.run(cmd, capture_output=True, text=True)
        res = res.stdout.split("\n")
        res = [re.split(r' +', s.strip()) for s in res]
        tags = res[0]
        res = [[int(c) if c.isnumeric() else c for c in r ] for r in res[2:-1]]
        self.info['img'] = dict()
        self.info['img']['count']=len(res)
        self.info['img']['area'] = sum([r[3]*r[4] for r in res])
        self.info['img']['frontpage'] = len(res)>0 and res[0][1]==1 and res[0][4]>1000
        self.info['img']['all'] = dict(zip(tags, list(map(list, zip(*res)))))

    def _get_meta_info(self):
        ''' runs pdfinfo, and stores the result as a dict '''
        cmd=["pdfinfo", "-isodates", self.filename]
        res = subprocess.run(cmd, capture_output=True, text=True)
        res = res.stdout.split("\n")
        res = [re.split(r': *',r.strip(), maxsplit=1) for r in res if len(r)>0]
        # Convert yes/no to bool:
        res = dict([(k, (v=='yes') if v in ('yes','no') else v) for k,v in res])
        self.info['meta'] = dict()
        self.info['meta']['touch_date'] = res.get('ModDate') or res.get('CreationDate')
        self.info['meta']['page_count'] = res.get('Pages')
        self.info['meta']['tagged'] = res.get('Tagged')
        return self.info

    def _is_scanned(self):
        ''' Heuristic test for a scanned document.
        If it only contains images, and no text, it is assumed to be scanned.'''
        cmd=["pdffonts", self.filename]
        res = subprocess.run(cmd, capture_output=True, text=True)
        res = res.stdout.split("\n")
        if not self.info.get('img'):
            self._img_info()
        self.info['scanned'] = (len(res)==2 and self.info["img"]["count"]>0)

class Node:
    ''' Helper class to construct a tree from the pdfinfo output '''
    def __init__(self, p,s):
        self.parent=p
        self.content=s
        self.children=[]
        self.is_text=s.strip()[:1] == '"'
        self.level=-1

    def as_html(self):
        ret = self.content.replace(":","").upper().split(" ")[0]
        ret=ret.replace("DOCUMENT","BODY")
        if ret[:1]=='/':
            ret="PRE"
        return ret
        
    def add_child(self, n):
        self.children.append(n)

    def build(self, s):
        if self.is_text:
            s.append(self.content[1:-1])
        else:
            s.append(f"<{self.as_html()}>")
        for c in self.children:
            s=c.build(s)
        if not self.is_text:
            s.append(f"</{self.as_html()}>")
        return s
    
    def __repr__(self):
        return f"level: {self.level}\ncontent: {self.content}\nparent: {self.parent.content}\nchildren: {len(self.children)>0}"

class DocxExtraction:
    ''' Convert docx documents to plain text.
    Basically a wrapper around pandoc.
    '''
    def __init__(self, docx_input):
        ''' Takes either a filename, a file handle, or a byte array as input. '''
        if type(docx_input) == io.BufferedReader:
            self.docx = docx_input.read()
        if type(docx_input) == bytes:
            self.docx = docx_input
        if type(docx_input) == str:
            with open(docx_input,"rb") as fh:
                self.docx = fh.read()

    def parse(self, output="plain"):
        ''' Runs pandoc on the docx document, and returns the result.
        Default output is plain, can be changed to anything pandoc can output,
        and which can be sent to STDOUT.'''
        cmd = ["pandoc", "-f", "docx", "-t", output]
        res = subprocess.run(cmd,
                             input=self.docx,
                             stdout=subprocess.PIPE)
        txt=res.stdout.decode("utf-8")
        return txt


    
if __name__ == "__main__":
    ''' Basic testing.
    If no pdf file given as arg, creates a basic one with
    pandoc at /tmp/test.pdf
    '''
    # Create test files
    pdf_test_file = '/tmp/test.pdf'  
    cmd = ['echo Hello World | pandoc -o '+pdf_test_file]
    res = subprocess.run(cmd, shell = True,executable="/bin/bash")
    docx_test_file = '/tmp/test.docx'  
    cmd = ['echo Hello World | pandoc -o '+docx_test_file]
    res = subprocess.run(cmd, shell = True,executable="/bin/bash")

    # PDF test
    test = PdfExtraction(pdf_test_file)
    test.run().cleanup()

    print(test.info)
    print("===============================================")
    print(test.text[:1000])
    print("===============================================")
    print(test.xml[:1000])
    print("===============================================")
    print(test.html[:1000])
    with open("/tmp/test.png","wb") as fh:
        fh.write(test.img)

    # DOCX test
    test=DocxExtraction(docx_test_file)
    print(test.parse())

    
