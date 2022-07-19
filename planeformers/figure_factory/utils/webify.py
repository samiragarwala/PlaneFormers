import shutil
from glob import glob
import dominate
from dominate.tags import *
from dominate.util import raw
import os
import datetime
import sys
import argparse
import math


IMG_EXT = ["png", "jpg", "jpeg", "gif", "tiff"]
VID_EXT = ["mp4", "avi", "mov", "webm"]
AUDIO_EXT = ['mp3']
MESH_EXT = ['glb', 'gltf']
is_image = lambda fn: any((fn.endswith(end) for end in IMG_EXT))
is_video = lambda fn: any((fn.endswith(end) for end in VID_EXT))
is_audio = lambda fn: any((fn.endswith(end) for end in AUDIO_EXT))
is_mesh = lambda fn: any((fn.endswith(end) for end in MESH_EXT))

page_width = 1500


def remove_prefix(text, prefix):
    text = text[text.startswith(prefix) and len(prefix):]
    text = text[text.startswith('/') and len('/'):]
    return text


def generate_page(prefix, filenames, page_id, num_pages, perrow=1, perpage=None, verbose=False):
    # scan files
    has_mesh = False
    has_video = False
    for filename in filenames:
        if is_video(filename):
            has_video = True
        if is_mesh(filename):
            has_mesh = True
        
    # compute width
    max_width = int(page_width / perrow)

    # base_id 
    if perpage is None:
        base_id = 0
    else:
        base_id = page_id * perpage

    # generate html
    doc = dominate.document(title='visualizer')
    with doc.head:
        # default layout
        link(rel='stylesheet', href='https://fouheylab.eecs.umich.edu/~jinlinyi/static/styles.css')
        
        # videojs
        if has_video:
            link(rel='stylesheet', href='https://fouheylab.eecs.umich.edu/~jinlinyi/static/videojs/video-js.css')
            link(rel='stylesheet', href='https://fouheylab.eecs.umich.edu/~jinlinyi/static/videojs/theme.css')
            script(type='text/javascript', src='https://fouheylab.eecs.umich.edu/~jinlinyi/static/videojs/video.js')
        
        # model-viewer
        if has_mesh:
            script(type='module', src='https://unpkg.com/@google/model-viewer/dist/model-viewer.js')

    # write prefix
    doc.add(p(prefix))

    # write pages
    if num_pages > 1:
        page_header = div()
        page_header.add('Pages: ')
        for link_page_id in range(num_pages):
            if link_page_id == page_id:
                page_header.add('{}'.format(link_page_id))
                page_header.add(raw('&nbsp;'))
            else:
                if link_page_id == 0:
                    page_link = a(href='view.html')
                else:
                    page_link = a(href='view_{}.html'.format(link_page_id))
                #page_link.add('Page {} '.format(link_page_id))
                page_link.add('{}'.format(link_page_id))
                #page_link.add(raw('&nbsp;'))
                page_header.add(page_link)
                page_header.add(raw('&nbsp;'))
        doc.add(page_header)

    for i, filename in enumerate(filenames):
        # remove prefix
        filename = remove_prefix(filename, prefix)

        # set up new row
        if i % perrow == 0:
            curr_row = div(cls='row')
            doc.add(curr_row)

        # create a column for each file
        curr_column = div(cls='column', style='width:{}px;max-height:1000px'.format(max_width))
        curr_column.add(p("{} -- {}".format(i + base_id, filename)))
        if is_image(filename):
            curr_column.add(img(src=filename, style='max-width:{}px;max-height:1000px'.format(max_width)))
        elif is_video(filename):
            # html5 style player
            #curr_column.add(video(src=filename, style='max-width:{}px;max-height:1000px'.format(max_width), controls=True))
            # videojs player
            vid = video(data_setup='{ "playbackRates": [0.25, 0.5, 1, 1.5, 2, 4, 8], "fluid": true }', cls="video-js vjs-default-skin", controls=True, preload='auto')
            vid.add(source(src=filename, type="video/mp4"))
            curr_column.add(vid)
        elif is_audio(filename):
            raise NotImplementedError("audio is not supported right now.")
        elif is_mesh(filename):
            mesh_vis = div()
            mesh_vis.add(raw('<model-viewer exposure="3" environment-image="neutral" interaction-prompt="when-focused" src="{}" shadow-intensity="1" camera-controls>'.format(filename)))
            curr_column.add(mesh_vis)
        else:
            raise NotImplementedError("cannot recognize {}".format(filename))
        curr_row.add(curr_column)

    # write pages again
    if num_pages > 1:
        page_header = div()
        page_header.add('Pages: ')
        for link_page_id in range(num_pages):
            if link_page_id == page_id:
                page_header.add('{}'.format(link_page_id))
                page_header.add(raw('&nbsp;'))
            else:
                if link_page_id == 0:
                    page_link = a(href='view.html')
                else:
                    page_link = a(href='view_{}.html'.format(link_page_id))
                #page_link.add('Page {} '.format(link_page_id))
                page_link.add('{}'.format(link_page_id))
                #page_link.add(raw('&nbsp;'))
                page_header.add(page_link)
                page_header.add(raw('&nbsp;'))
        doc.add(page_header)

    doc.add(p("Made at {}".format(datetime.datetime.now())))

    # write
    html_name = 'view_{}.html'.format(page_id) if page_id != 0 else 'view.html'
    html_file = os.path.join(prefix, html_name)
    f = open(html_file, 'wt')
    f.write(doc.render())
    f.close()


def generate_webpage(prefix, regex, perrow=1, perpage=None, verbose=False):
    """
    Generate a webpage to prefix/view.html, based on prefix/regex.
    perpage: number of items per row.
    perpage: number of rows per page.
    verbose: print information
    """
    # video files
    filenames = []
    regex_list = regex.split('|')
    for regex_single in regex_list:
        filenames.extend(glob(os.path.join(prefix, regex_single)))
    filenames.sort()
    if verbose:
        print('find {} files'.format(len(filenames)))

    # split pages
    if perpage is None:
        num_pages = 1
    else:
        num_pages = math.ceil(len(filenames) / perpage)

    if num_pages == 1:  # single page
        generate_page(prefix, filenames, 0, num_pages=1, perrow=perrow, perpage=perpage, verbose=verbose)
    else:  # multi page
        filenames = [filenames[i:i + perpage] for i in range(0, len(filenames), perpage)]
        for page_id in range(num_pages):
            page_filenames = filenames[page_id]
            generate_page(prefix, page_filenames, page_id, num_pages=num_pages, perrow=perrow, perpage=perpage, verbose=verbose)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('prefix', type=str)
    parser.add_argument('regex', type=str)
    parser.add_argument('--column', type=int, default=1)
    parser.add_argument('--perpage', type=int, default=None)
    
    args = parser.parse_args()
    return args
    

def main():
    """
    Usage: python webify.py /path/to/folder "*.png"
    """
    args = parse_args()
    generate_webpage(args.prefix, args.regex, args.column, args.perpage, verbose=True)


if __name__=='__main__':
    main()
