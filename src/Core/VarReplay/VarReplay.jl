module VarReplay

import Base: string, isequal, ==, hash, getindex, setindex!, push!, show, isempty

using ...Turing: CACHERESET, CACHEIDCS, CACHERANGES
using ...Samplers
using Distributions
using Parameters: @unpack
using ...Utilities
import ...Utilities: flatten
using Bijectors

export  VarName, 
        VarInfo, 
        UntypedVarInfo, 
        TypedVarInfo, 
        AbstractVarInfo, 
        uid, 
        sym, 
        getlogp, 
        set_retained_vns_del_by_spl!, 
        resetlogp!, 
        is_flagged, 
        unset_flag!, 
        setgid!, 
        copybyindex, 
        setorder!, 
        updategid!, 
        acclogp!, 
        istrans, 
        link!, 
        invlink!, 
        setlogp!, 
        getranges, 
        getrange, 
        getvns, 
        getval,
        NewVarInfo

include("varinfo.jl")
include("typed_varinfo.jl")

@generated function flatten(names, value :: Array{Float64}, k :: String, v::TypedVarInfo{Tvis}) where Tvis
    expr = Expr(:block)
    for f in fieldnames(Tvis)
        push!(expr.args, quote
            idcs = v.vis.$f.idcs
            ranges = v.vis.$f.ranges
            vals = v.vis.$f.vals
            for (vn, i) in idcs
                range = ranges[i]
                flatten(names, value, string(sym(vn)), vals[range])
            end
        end)
    end
    return expr
end
function flatten(names, value :: Array{Float64}, k :: String, v::UntypedVarInfo)
    idcs = v.idcs
    ranges = v.ranges
    vals = v.vals
    for (vn, i) in idcs
        range = ranges[i]
        flatten(names, value, string(sym(vn)), vals[range])
    end
end

end # module
