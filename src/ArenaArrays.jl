module ArenaArrays

using LinearAlgebra

export AbstractArenaArray
export ArenaArray
export ArenaWrappedArray

abstract type AbstractArenaArray{T, N} <: DenseArray{T, N} end

Base.IndexStyle(::Type{<:AbstractArenaArray}) = IndexLinear()

Base.eltype(::AbstractArenaArray{T}) where T = T

Base.elsize(::AbstractArenaArray{T}) where T = sizeof(T)

struct ArenaArray{T, N} <: AbstractArenaArray{T, N}
    arena::Vector{UInt8}
    offset::Int
    dims::NTuple{N, Int}
end

function ArenaArray{T}(arena::Vector{UInt8}, dims::NTuple{N, Int}) where {T, N}
    offset = length(arena)
    s = prod(dims) * sizeof(T)
    resize!(arena, offset + s)
    ArenaArray{T, N}(arena, offset, dims)
end

function ArenaArray(arena::Vector{UInt8}, src::AbstractArray{T, N}) where {T, N}
    dst = ArenaArray{T}(arena, size(src))
    copy!(dst, src)
    dst
end

@inline function Base.unsafe_convert(::Type{Ptr{T}}, a::ArenaArray{T}) where T
    Base.unsafe_convert(Ptr{T}, a.arena) + a.offset
end

Base.size(a::ArenaArray) = a.dims

@inline Base.dataids(a::ArenaArray) = (UInt(pointer(a)),)

@inline function Base.getindex(a::ArenaArray, i::Integer)
    @boundscheck checkbounds(a, i)
    GC.@preserve a unsafe_load(pointer(a), i)
end

@inline function Base.setindex!(a::ArenaArray, x, i::Integer)
    @boundscheck checkbounds(a, i)
    GC.@preserve a unsafe_store!(pointer(a), x, i)
end

@inline function Base.similar(a::AbstractArenaArray{T, N}, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    ArenaArray{T}(a.arena, dims)
end

@inline function Base.reshape(a::ArenaArray{T, M}, dims::NTuple{N, Int}) where {T, M, N}
    ArenaArray{T, N}(a.arena, a.offset, dims)
end

Base.BroadcastStyle(::Type{<:AbstractArenaArray}) = Broadcast.ArrayStyle{ArenaArray}()

# TODO: this is from the example in the manual, doesn't seem super clean.
#       try to optimize and in particular force type stability on this
find_arena_(bc::Broadcast.Broadcasted) = find_arena_(bc.args)
find_arena_(args::Tuple) = find_arena_(find_arena_(args[1]), Base.tail(args))
find_arena_(x) = x
find_arena_(::Tuple{}) = nothing
find_arena_(a::AbstractArenaArray, rest) = a
find_arena_(::Any, rest) = find_arena_(rest)

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ArenaArray}}, ::Type{T}) where T
    isbitstype(T) || return Array{T}(size(bc))
    arena = find_arena_(bc).arena::Vector{UInt8}
    ArenaArray{T}(arena, size(bc))
end

struct ArenaWrappedArray{T, N, A} <: AbstractArenaArray{T, N}
    arena::Vector{UInt8}
    wrapped::A
    
    ArenaWrappedArray(arena::Vector{UInt8}, wrapped::DenseArray{T, N}) where {T, N} = new{T, N, typeof(wrapped)}(
        arena, wrapped
    )
end

@inline Base.dataids(a::ArenaWrappedArray) = Base.dataids(a.wrapped)

@inline function Base.unsafe_convert(::Type{Ptr{T}}, a::ArenaWrappedArray{T}) where T
    Base.unsafe_convert(Ptr{T}, a.wrapped)
end

Base.size(a::ArenaWrappedArray) = size(a.wrapped)

Base.@propagate_inbounds Base.getindex(a::ArenaWrappedArray, i...) = getindex(a.wrapped, i...)

# 
# This is hardly a satisfying solution, but Base only ever looks the rhs operand to determine result type
# so to ensure matmul results are ArenaArray objects we implement the full suite of overloads
# Further aggravating, the overloads need to be sufficiently specific that they don't cause
# method ambiguities. Would love to see a better solution to this
#

@inline function Base.:(*)(A::AbstractArenaArray{T, 2}, B::Matrix{T}) where T
    TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    mul!(similar(A, TS, (size(A,1), size(B,2))), A, B)
end

@inline function Base.:(*)(A::Matrix{T}, B::AbstractArenaArray{T, 2}) where T
    TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    mul!(similar(B, TS, (size(A,1), size(B,2))), A, B)
end

@inline function Base.:(*)(A::AbstractArenaArray{T, 2}, B::AbstractArenaArray{T, 2}) where T
    TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    mul!(similar(B, TS, (size(A,1), size(B,2))), A, B)
end

for L in (true, false)
    @eval @inline function Base.:(*)(A::SubArray{T, 2, <:AbstractArenaArray{T, 2}, I, $L}, B::Matrix{T}) where {T, I}
        TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
        mul!(similar(A, TS, (size(A,1), size(B,2))), A, B)
    end

    @eval @inline function Base.:(*)(A::Matrix{T}, B::SubArray{T, 2, <:AbstractArenaArray{T, 2}, I, $L}) where {T, I}
        TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
        mul!(similar(B, TS, (size(A,1), size(B,2))), A, B)
    end
end

for L1 in (true, false)
    for L2 in (true, false)
        @eval @inline function Base.:(*)(
            A::SubArray{T, 2, <:AbstractArenaArray{T, 2}, I, $L1},
            B::SubArray{T, 2, <:AbstractArenaArray{T, 2}, I, $L2}
        ) where {T, I}
            TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
            mul!(similar(B, TS, (size(A,1), size(B,2))), A, B)
        end
    end
end

end
