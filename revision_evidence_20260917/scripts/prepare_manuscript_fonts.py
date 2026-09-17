"""Convert local Libertine CFF outlines to TrueType for reliable Matplotlib PDF embedding."""
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
HERE=Path(__file__).resolve().parents[1]
SOURCE=Path.home()/'AppData/Local/Programs/MiKTeX/fonts/opentype/public/libertine'
OUT=HERE/'assets/fonts';OUT.mkdir(parents=True,exist_ok=True)
for stem,style,bold,italic in [('R','Regular',False,False),('RB','Bold',True,False),('RI','Italic',False,True),('RBI','Bold Italic',True,True)]:
    old=TTFont(SOURCE/f'LinLibertine_{stem}.otf');order=old.getGlyphOrder();glyphset=old.getGlyphSet();glyphs={}
    for name in order:
        pen=TTGlyphPen(glyphset);converter=Cu2QuPen(pen,max_err=1.0,reverse_direction=True)
        glyphset[name].draw(converter);glyphs[name]=pen.glyph()
    fb=FontBuilder(old['head'].unitsPerEm,isTTF=True)
    fb.setupGlyphOrder(order);fb.setupCharacterMap(old.getBestCmap());fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(old['hmtx'].metrics)
    fb.setupHorizontalHeader(ascent=old['hhea'].ascent,descent=old['hhea'].descent,lineGap=old['hhea'].lineGap)
    fb.setupOS2(sTypoAscender=old['OS/2'].sTypoAscender,sTypoDescender=old['OS/2'].sTypoDescender,
                usWinAscent=old['OS/2'].usWinAscent,usWinDescent=old['OS/2'].usWinDescent,
                usWeightClass=700 if bold else 400,fsSelection=(1 if italic else 0)|(32 if bold else 0)|(64 if not bold and not italic else 0))
    fb.setupNameTable({'familyName':'Manuscript Libertine','styleName':style,
       'uniqueFontIdentifier':'ManuscriptLibertine-'+style.replace(' ',''),
       'fullName':'Manuscript Libertine '+style,'psName':'ManuscriptLibertine-'+style.replace(' ',''),
       'version':'Version 1.0; local Libertine outline conversion for this evidence package',
       'copyright':old['name'].getDebugName(0) or '',
       'licenseDescription':old['name'].getDebugName(13) or 'Original Linux Libertine license applies.',
       'licenseInfoURL':old['name'].getDebugName(14) or ''})
    fb.setupPost(italicAngle=old['post'].italicAngle);fb.setupMaxp()
    fb.font['head'].macStyle=(1 if bold else 0)|(2 if italic else 0)
    fb.save(OUT/f'ManuscriptLibertine-{stem}.ttf')
    print('FONT',stem,flush=True)
