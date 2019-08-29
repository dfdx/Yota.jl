@diffrule_kw conv(x, w, _cdims) x ∇conv_data(dy, w, _cdims)
@diffrule_kw conv(x, w, _cdims) w ∇conv_filter(x, dy, _cdims)
@nodiff conv(x, w, _cdims) _cdims

@diffrule_kw maxpool(x, _pdims) u ∇maxpool(dy, y, x, _pdims)
@nodiff maxpool(x, _pdims) _pdims
