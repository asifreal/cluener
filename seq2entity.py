def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_io(seq, id2label):
    """Gets entities from sequence.
    note: IO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['I-PER', 'I-PER', 'O', 'I-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    if not isinstance(seq[0], str):
        inp = []
        for i in range(len(seq)):
            inp.append(id2label[seq[i]])
        seq = inp
    i, j = 0, 0
    while i < len(seq):
        if seq[i].startswith('I'):
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j+=1
            chunk = [seq[i].split('-')[1], i, j-1]
            chunks.append(chunk)
            i = j
        else:
            i+=1
    return chunks

def get_entity_oi(seq, id2label):
    """Gets entities from sequence.
    note: IO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['I', 'I', 'O', 'I']
        get_entity_bio(seq)
        #output
        [['', 0,1], ['', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    if not isinstance(seq[0], str):
        inp = []
        for i in range(len(seq)):
            inp.append(id2label[seq[i]])
        seq = inp
    i, j = 0, 0
    while i < len(seq):
        if seq[i].startswith('I'):
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j+=1
            chunk = ['', i, j-1]
            chunks.append(chunk)
            i = j
        else:
            i+=1
    return chunks

table = {
    'oi': get_entity_oi,
    'io': get_entity_io,
    'bio': get_entity_bio,
    'bios': get_entity_bios
}

def get_entities(seq,id2label,markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios', 'io','oi']
    return table[markup](seq,id2label)
