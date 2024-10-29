import pandas as pd
import numpy as np
import plotnine as p9
import squarify

__version__ = '0.21'

def tg_treemap(df, 
              width:int=1200, 
              height:int=800, 
              h1_pad:int=23, 
              h2_pad:int=0,
              pads:list=[0,2,1], 
              fillcolors = ['#4E79A7', '#A0CBE8', '#F28E2B', '#FFBE7D', '#59A14F', '#8CD17D', '#B6992D', '#F1CE63', '#499894', '#86BCB6', '#E15759', '#FF9D9A', '#79706E', '#BAB0AC', '#D37295', '#FABFD2', '#B07AA1', '#D4A6C8', '#9D7660', '#D7B5A6'], 
              fontcolors:list =['black','darkblue','black'],
              fontfamily = 'Arial',
              fontsizes:list = [12,11,10],
              labeltype:list = ['label','text','text'],
              includeNumbers: list =['no','no','no'],
              l2_mul_horizontally:float = 6,
              l3_mul_horizontally:float = 6,
              l2_space_none = 12,
              l3_space_none = 7,
              fillcolumn:str = 'h1',
              linecolors:list = ['black','black','darkgray'],
              figsize:list= [12*1.2,8*1.2],
              backgroundcolor:str = 'darkgray',
              canvascolor:str ='white',
              override:dict={}):

    """width   ... width of box where tree is drawn. This is a calculation basis, neither pixel nor cm.
            ... Breite des Kastens wo die treemap gezeichnet wird. Dient als Kalkulationsbasis, entspricht weder Pixel noch cm.
    height  ... width of box where tree is drawn. This is a calculation basis, neither pixel nor cm.
            ... Höhe des Kastens wo die treemap gezeichnet wird. Dient als Kalkulationsbasis, entspricht weder Pixel noch cm.
    h1_pad  ... Distance in h1 from top edge to the beginning of h2 
            ... Entfernung in h1 vom oberen Rand zu Beginn von h2
    h2_pad  ... Distance in h2 from top edge to the beginning of h3 
            ... Entfernung in h2 vom oberen Rand zu Beginn von h3
    pads    ... Distance between inner and outer boxes
            ... Entferneung zwischen inneren und äußeren Kästchen
    colors  ... fill colors for boxes (according to fillcolumn)
            ... Farben mit denen die Kästchen gefüllt werden (je nach fillcolumn)
    fontcolors ... Font colors for h1, h2, h3 as list
               ... Schriftfarben für h1, h2, h3 als Liste
    fontsizes ... Fontsize h1, h2, h3 as list
              ... Größe der Schriftarten von h1, h2, h3 als Liste
    labeltype ... either 'label' or 'text' according to 'geom_label' or 'geom_text'
                  entweder 'label' oder 'text'
    includeNumbers ... should numbers be included in text labels (at the end) for h1, h2, h3
                       'no',   no numbers 
                       'yes',  yes include numbers
                       'only', only show numbers 
                       as list
                   ... sollen die Werte für h1,h2,h3 in der Beschriftung dabei sein (am Ende) 
                       'no',   keine Werte 
                       'yes'   ja, dabei
                       'only'  nur die Werte, kein Text
    l2_mul_horizontally ... multiplier to determine when to change the text orientation of h2 from landscape to portrait, 
                            the number is multiplied with the number of characters and can seen as something like the width of a character, 
                            as lower as sooner the text orientation of h2 is changed from landscape to portrait 
                        ... Multiplikator zum Bestimmen wann von der h2 Text von Querformat auf Hochformat umgestellt wird,
                            die Zahl wird mit der Anzahl der Buchstaben multipliziert und kann und kann als Buchstabenbreite interpretiert werden,
                            je kleiner umso früher wird die Orientierung von h2 von Querformat auf Hochformat umgestellt 
    l3_mul_horizontally ... multiplier to determine when to change the text orientation of h3 from landscape to portrait, 
                            the number is multiplied with the number of characters and can seen as something like the width of a character, 
                            as lower as sooner the text orientation of h3 is changed from landscape to portrait 
                        ... Multiplikator zum Bestimmen wann von der h3 Text von Querformat auf Hochformat umgestellt wird,
                            die Zahl wird mit der Anzahl der Buchstaben multipliziert und kann und kann als Buchstabenbreite interpretiert werden,
                            je kleiner umso früher wird die Orientierung von h3 von Querformat auf Hochformat umgestellt 
    l2_space_none ... at what size of the box, both in x and y, is no label made in h2
                  ... bei welcher Größe des Kastens sowohl in x als auch in y wird keine Beschriftung in h2 gemacht
    l3_space_none ... At what size of the box, both in x and y, is no label made in h3
                  ... bei welcher Größe des Kastens sowohl in x als auch in y wird keine Beschriftung in h3 gemacht
    fillcolumn ... determins if h1, h2 or h3 is used to set colors
               ... welche Hierarchie wird zur Farbfüllung benutzt
    figsize ... Size of figure (plotnine or matplotlib.pyplot.figure) in inches
            ... Größe des Diagramms (plotnine oder matplotlib.pyplot.figure) in Zoll
    override ... items of h3 to be colored individually as dictionary
             ... Elemente von h3 die individuell eingefärbt werden sollen als dictionary
    backgroundcolor ... color for the entire canvas outside the plotting area
                    ... Farbe der Leinwand ausserhalb des Graphen
"""
    if override:
        override = pd.DataFrame(override).T.reset_index().rename(columns={'index':'h3'})

    if not 'value' in df.columns:
        return "no column 'value'"
    
    if not 'h1' in df.columns:
        return "no column 'h1'"

    if ((not 'h2' in df.columns) & ('h3' in df.columns)):
        df = df.rename(columns={'h3':'h2'})
        
    if not fillcolumn in df.columns:
        fillcolumn='h1'

    labeltype = ["text" if x != "text" and x != "label" else x for x in labeltype]
    if len(labeltype) <3:
        labeltype += ['text'] * (3 - len(labeltype))

    includeNumbers = [item if item in ['yes', 'no', 'only'] else 'no' for item in includeNumbers]
    if len(includeNumbers) <3:
        includeNumbers += ['no'] * (3 - len(includeNumbers))

    if len(linecolors) <3:
        linecolors += ['darkgray'] * (3 - len(linecolors))

    def positions(data, width, height, x=0, y=0, pad=0):
        data = data.reset_index(drop=True)
        norm = squarify.normalize_sizes(data['value'].to_list(), width, height)
        result = pd.DataFrame(squarify.squarify(norm, x=x,y=y, dx=width, dy=height))
        result = pd.concat([data, result], axis=1)
        result[['x','y','dx','dy']] = result[['x','y','dx','dy']].round(0)
        result['x'] = np.where(result['dx']>2*pad, result['x']+pad, result['x'])
        result['dx'] = np.where(result['dx']>2*pad, result['dx']-pad*2 , result['dx'])
        result['y'] = np.where(result['dy']>2*pad, result['y']+pad, result['y'])
        result['dy'] = np.where(result['dy']>2*pad, result['dy']-pad*2 , result['dy'])
        return result
    
    s1 = df.groupby('h1').agg({'value':'sum'}).reset_index()
    p1 = positions(data=s1,width=width,height=height, pad=pads[0])
    
    layer1 = p1.copy()
    beschriftung1 = layer1.copy()
    beschriftung1['stext'] = beschriftung1['h1']
    if includeNumbers[0] == 'yes':
        beschriftung1['stext'] = beschriftung1['stext'] + ' ' + beschriftung1['value'].astype('str')
    if includeNumbers[0] == 'only':
        beschriftung1['stext'] = beschriftung1['value'].astype('str')

    if 'h2' in df.columns:
        s2 = df.groupby(['h1','h2']).agg({'value':'sum'}).reset_index()
        layer2 = pd.DataFrame()
        for _, item in s1.iterrows():
            who = item['h1']
            df_werte = s2.query("h1 == @who")
            df_pos = layer1.query("h1 == @who")
            p2= positions(data=df_werte, width=df_pos['dx'].iloc[0], height=df_pos['dy'].iloc[0]-h1_pad, x=df_pos['x'].iloc[0], y=df_pos['y'].iloc[0], pad=pads[1])
            layer2 = pd.concat([layer2, p2]).reset_index(drop=True)
        layer2['space'] = layer2['h2'].str.len().astype('float').mul(l2_mul_horizontally)
        layer2['space'] = layer2['space'].round(0).astype('int')
        layer2['orientation'] = np.where(layer2['dx']<layer2['space'],'portrait','horizontally')
        layer2['orientation'] = np.where(((layer2['orientation']=='portrait') & 
                                          (layer2['dy']<layer2['space'])),'keine',layer2['orientation'])
        layer2['orientation'] = np.where(layer2['dy']<=l2_space_none,'keine',layer2['orientation'])
        layer2['orientation'] = np.where(layer2['dx']<=l2_space_none,'keine',layer2['orientation'])
    
        beschriftung2w = layer2.query("orientation == 'horizontally'").reset_index(drop=True)
        beschriftung2s = layer2.query("orientation == 'portrait'").reset_index(drop=True)
        beschriftung2w['stext'] = beschriftung2w['h2']
        beschriftung2s['stext'] = beschriftung2s['h2']
        
        if includeNumbers[1] == 'yes':
            beschriftung2w['stext'] = beschriftung2w['stext'] + ' ' + beschriftung2w['value'].astype('str')
            beschriftung2s['stext'] = beschriftung2s['stext'] + ' ' + beschriftung2s['value'].astype('str')
        if includeNumbers[1] == 'only':
            beschriftung2w['stext'] = beschriftung2w['value'].astype('str')
            beschriftung2s['stext'] = beschriftung2s['value'].astype('str')

    if 'h3' in df.columns:
        s3 = df.groupby(['h1','h2','h3']).agg({'value':'sum'}).reset_index()
        layer3 = pd.DataFrame()
       
        for _, h1 in s1.iterrows():
            who_h1 = h1['h1']
            for _, h2 in s2.iterrows():
                who_h2 = h2['h2']
                df_werte = s3.query("h2 == @who_h2").query("h1 == @who_h1")
                df_pos = layer2.query("h2 == @who_h2").query("h1 == @who_h1")
                if df_pos.shape[0]>0:
                    p3= positions(data=df_werte, width=df_pos['dx'].iloc[0], height=df_pos['dy'].iloc[0]-h2_pad, x=df_pos['x'].iloc[0], y=df_pos['y'].iloc[0], pad=pads[2])
                    layer3 = pd.concat([layer3, p3]).reset_index(drop=True)
        
        layer3['space'] = layer3['h3'].str.len().astype('float').mul(l3_mul_horizontally)
        layer3['space'] = layer3['space'].round(0).astype('int')
        layer3['orientation'] = np.where(layer3['dx']<layer3['space'],'portrait','horizontally')
        layer3['orientation'] = np.where(((layer3['orientation']=='portrait') & 
                                          (layer3['dy']<layer3['space'])),'keine',layer3['orientation'])
        layer3['orientation'] = np.where(layer3['dy']<=l3_space_none,'keine',layer3['orientation'])
        layer3['orientation'] = np.where(layer3['dx']<=l3_space_none,'keine',layer3['orientation'])
        beschriftung3w = layer3.query("orientation == 'horizontally'").reset_index(drop=True)
        beschriftung3s = layer3.query("orientation == 'portrait'").reset_index(drop=True)
        
        beschriftung3w['stext'] = beschriftung3w['h3']
        beschriftung3s['stext'] = beschriftung3s['h3']

        if includeNumbers[2] == 'yes':
            beschriftung3w['stext'] = beschriftung3w['stext'] + ' ' + beschriftung3w['value'].astype('str')
            beschriftung3s['stext'] = beschriftung3s['stext'] + ' ' + beschriftung3s['value'].astype('str')
        if includeNumbers[2] == 'only':
            beschriftung3w['stext'] = beschriftung3w['value'].astype('str')
            beschriftung3s['stext'] = beschriftung3s['value'].astype('str')

        #override
        if isinstance(override, pd.DataFrame) and not override.empty:
            beschriftung3wo = beschriftung3w[beschriftung3w['h3'].isin(override['h3'].to_list())]
            beschriftung3wn = beschriftung3w[~beschriftung3w['h3'].isin(override['h3'].to_list())]
            beschriftung3so = beschriftung3s[beschriftung3s['h3'].isin(override['h3'].to_list())]
            beschriftung3sn = beschriftung3s[~beschriftung3s['h3'].isin(override['h3'].to_list())]

            if beschriftung3wo.shape[0]>0:
                beschriftung3wo = beschriftung3wo.merge(override, on='h3', how='left')
                if 'fontcolor' in beschriftung3wo.columns:
                    beschriftung3wo['fontcolor'] = beschriftung3wo['fontcolor'].fillna(fontcolors[2])
                else:
                    beschriftung3wo['fontcolor'] = fontcolors[2]

                if 'fontsize' in beschriftung3wo.columns:
                    beschriftung3wo['fontsize'] = beschriftung3wo['fontsize'].fillna(fontsizes[2])
                else:
                    beschriftung3wo['fontsize'] = fontsizes[2]

                if 'fontfamily' in beschriftung3wo.columns:
                    beschriftung3wo['fontfamily'] = beschriftung3wo['fontfamily'].fillna(fontfamily)
                else:
                    beschriftung3wo['fontfamily'] = fontfamily

            if beschriftung3so.shape[0]>0:
                beschriftung3so = beschriftung3so.merge(override, on='h3', how='left')
                if 'fontcolor' in beschriftung3so.columns:
                    beschriftung3so['fontcolor'] = beschriftung3so['fontcolor'].fillna(fontcolors[2])
                else:
                    beschriftung3so['fontcolor'] = fontcolors[2]

                if 'fontsize' in beschriftung3so.columns:
                    beschriftung3so['fontsize'] = beschriftung3so['fontsize'].fillna(fontsizes[2])
                else:
                    beschriftung3so['fontsize'] = fontsizes[2]

                if 'fontfamily' in beschriftung3wo.columns:
                    beschriftung3so['fontfamily'] = beschriftung3so['fontfamily'].fillna(fontfamily)
                else:
                    beschriftung3so['fontfamily'] = fontfamily
        else:
            beschriftung3wo = pd.DataFrame()
            beschriftung3wn = beschriftung3w.copy()
            beschriftung3so = pd.DataFrame()
            beschriftung3sn = beschriftung3s.copy()


        layer3 = layer3.merge(df.drop('value',axis=1),on=['h1','h2','h3'], how='left')

    #boxes 
    
    p=(p9.ggplot())
    p =p + p9.geom_rect(mapping=p9.aes(xmin=0,xmax=width-1, ymin=0, ymax=height-1) ,fill=backgroundcolor, color=linecolors[0])

    if fillcolumn=='h1':
        p=(p
        +p9.geom_rect(data= layer1, mapping=p9.aes(xmin='x',xmax='x+dx', ymin='y', ymax='y+dy', fill='h1'), color=linecolors[0])
        )
    else:
        p=(p
        +p9.geom_rect(data= layer1, mapping=p9.aes(xmin='x',xmax='x+dx', ymin='y', ymax='y+dy'), fill=None, color=linecolors[0])
        )

    if 'h2' in df.columns:
       if fillcolumn=='h2':
        p = (p
           +p9.geom_rect(data= layer2, mapping=p9.aes(xmin='x',xmax='x+dx', ymin='y', ymax='y+dy', fill='h2'), color=linecolors[1])
            )
        pass
       else:
           p = (p
           +p9.geom_rect(data= layer2, mapping=p9.aes(xmin='x',xmax='x+dx', ymin='y', ymax='y+dy'), fill=None, color=linecolors[1])
            )


    if 'h3' in df.columns:
        if fillcolumn=='h1':
            p=(p
            +p9.geom_rect(data= layer3, mapping=p9.aes(xmin='x',xmax='x+dx', ymin='y', ymax='y+dy', fill=fillcolumn), color=linecolors[2])
            )
        else:
            p=(p
            +p9.geom_rect(data= layer3, mapping=p9.aes(xmin='x',xmax='x+dx', ymin='y', ymax='y+dy', fill=fillcolumn), color=linecolors[1])
            )
    
        if isinstance(override, pd.DataFrame) and not override.empty:
            if type(override)==dict:
                override = pd.DataFrame(override).T.reset_index().rename(columns={'index':'h3'})

            if 'fillcolor' in override.columns:
                ovr_fill = layer3.merge(override, on='h3', how='left').dropna(subset='fillcolor').reset_index(drop=True)
                for i in range(0, ovr_fill.shape[0]):
                    temp = ovr_fill[i:i+1]
                    ocolor = temp['fillcolor'].iloc[0]
                    p = p + p9.geom_rect(data=temp, mapping=p9.aes(xmin='x', xmax='x+dx', ymin='y', ymax='y+dy'), fill=ocolor)


    p = p + p9.geom_rect(mapping=p9.aes(xmin=0,xmax=width-1, ymin=0, ymax=height-1),fill=None, color=linecolors[0])
    
    # labels
    if labeltype[0]=='label':
        p=p+p9.geom_label(data= beschriftung1, mapping=p9.aes(x='x+6', y='y+dy-6', label='stext'), 
                           ha='left', va = 'top', 
                           family=fontfamily, 
                           size=fontsizes[0], 
                           color=fontcolors[0])
    else:
        p=p+p9.geom_text(data= beschriftung1, mapping=p9.aes(x='x+3', y='y+dy-3', label='stext'), 
                           ha='left', va = 'top', 
                           family=fontfamily, 
                           size=fontsizes[0], 
                           color=fontcolors[0])
    
    if 'h2' in df.columns:
        if labeltype[1]=='label':
            p = (p
            +p9.geom_label(data= beschriftung2w, mapping=p9.aes(x='x+2', y='y+dy-3', label='stext'), 
                        ha='left', va = 'top', 
                        family=fontfamily, 
                        size=fontsizes[1], 
                        color= fontcolors[1])
            +p9.geom_label(data= beschriftung2s, mapping=p9.aes(x='x', y='y+dy', label='stext'), 
                        ha='left', va = 'top', 
                        family=fontfamily, 
                        size=fontsizes[1], 
                        color = fontcolors[1],
                        angle=90)
            )
        else:
            p = (p
            +p9.geom_text(data= beschriftung2w, mapping=p9.aes(x='x+2', y='y+dy-3', label='stext'), 
                        ha='left', va = 'top', 
                        family=fontfamily, 
                        size=fontsizes[1], 
                        color= fontcolors[1])
            +p9.geom_text(data= beschriftung2s, mapping=p9.aes(x='x', y='y+dy', label='stext'), 
                        ha='left', va = 'top', 
                        family=fontfamily, 
                        size=fontsizes[1], 
                        color = fontcolors[1],
                        angle=90)
            )

    if 'h3' in df.columns:
        if labeltype[2]=='label':
            p=(p
            +p9.geom_label(data= beschriftung3wn, mapping=p9.aes(x='x+dx/2', y='y+dy/2', label='stext'), 
                        ha='center', va = 'center', 
                        family=fontfamily, 
                        size=fontsizes[2], 
                        color =fontcolors[2])
            +p9.geom_label(data= beschriftung3sn, mapping=p9.aes(x='x+dx/2', y='y+dy/2', label='stext'), 
                       ha='center', va = 'center', 
                       family=fontfamily, 
                       size=fontsizes[2], 
                       color =fontcolors[2], 
                       angle=90)
                )
            #override
            if beschriftung3wo.shape[0]>0:
                for i in range(0, beschriftung3wo.shape[0]):
                    temp = beschriftung3wo[i:i+1]
                    fcolor = temp['fontcolor'].iloc[0]
                    ffamily = temp['fontfamily'].iloc[0]
                    fsize = temp['fontsize'].iloc[0]
                    p = p + p9.geom_label(data= temp, mapping=p9.aes(x='x+dx/2', y='y+dy/2', label='stext'), 
                            ha='center', va = 'center', 
                            family=ffamily, 
                            size=fsize, 
                            color =fcolor)
            if beschriftung3so.shape[0]>0:
                for i in range(0, beschriftung3so.shape[0]):
                    temp = beschriftung3so[i:i+1]
                    fcolor = temp['fontcolor'].iloc[0]
                    ffamily = temp['fontfamily'].iloc[0]
                    fsize = temp['fontsize'].iloc[0]
                    p = p + p9.geom_label(data= temp, mapping=p9.aes(x='x+dx/2', y='y+dy/2', label='stext'), 
                            ha='center', va = 'center', 
                            family=ffamily, 
                            size=fsize, 
                            color =fcolor,
                            angle=90)
        else:
            p=(p
            +p9.geom_text(data= beschriftung3wn, mapping=p9.aes(x='x+dx/2', y='y+dy/2', label='stext'), 
                        ha='center', va = 'center', 
                        family=fontfamily, 
                        size=fontsizes[2], 
                        color =fontcolors[2])
            +p9.geom_text(data= beschriftung3sn, mapping=p9.aes(x='x+dx/2', y='y+dy/2', label='stext'), 
                        ha='center', va = 'center', 
                        family=fontfamily, 
                        size=fontsizes[2], 
                        color =fontcolors[2], 
                        angle=90)
            )
            #override
            if beschriftung3wo.shape[0]>0:
                for i in range(0, beschriftung3wo.shape[0]):
                    temp = beschriftung3wo[i:i+1]
                    fcolor = temp['fillcolor'].iloc[0]
                    ffamily = temp['fontfamily'].iloc[0]
                    fsize = temp['fontsize'].iloc[0]
                    p = p + p9.geom_text(data= temp, mapping=p9.aes(x='x+dx/2', y='y+dy/2', label='stext'), 
                            ha='center', va = 'center', 
                            family=ffamily, 
                            size=fsize, 
                            color =fcolor)
            if beschriftung3so.shape[0]>0:
                for i in range(0, beschriftung3so.shape[0]):
                    temp = beschriftung3so[i:i+1]
                    fcolor = temp['fillcolor'].iloc[0]
                    ffamily = temp['fontfamily'].iloc[0]
                    fsize = temp['fontsize'].iloc[0]
                    p = p + p9.geom_text(data= temp, mapping=p9.aes(x='x+dx/2', y='y+dy/2', label='stext'), 
                            ha='center', va = 'center', 
                            family=ffamily, 
                            size=fsize, 
                            color =fcolor,
                            angle=90)
    p=(p
   
    +p9.scale_fill_manual(values=fillcolors)
    +p9.theme_minimal()
    +p9.coord_fixed()
    +p9.labs(x='',y='')
    +p9.theme(figure_size=figsize)
    +p9.theme(legend_position='none',
            legend_title=p9.element_blank())
    +p9.theme(axis_ticks=p9.element_blank(), panel_grid=p9.element_blank(), axis_text=p9.element_blank())
    +p9.theme(plot_background=p9.element_rect(fill=canvascolor))
    +p9.theme(panel_background=p9.element_rect(fill=canvascolor))
    
    )

    return p, layer3
