using GLMakie
using StaticArrays

module SimulationBuilder

    using GLMakie
    using LinearAlgebra

    const I = Int64
    const S = Float64

    @enum PlacementMode ModifyMode VertexMode TriangleMode ActuatorMode 

    mutable struct BuilderState
        vertices::Matrix{S}
        indices::Vector{I}
        actuators::Vector{Tuple{I,I}}
        mode::PlacementMode
        mode_state::Any
    end

    function create()
        state = Observable(BuilderState(Matrix{S}(undef, 2,0), Vector{I}(), Vector{Tuple{I,I}}(), ModifyMode, ()))

        fig = Figure()

        ax = Axis(
            fig[1,1], 
            aspect = 1, 
            limits = (-1., 1., -0.1, 1.9), 
            title = lift(state) do st
                if st.mode == ModifyMode
                    "Normal"
                elseif st.mode == VertexMode
                    "Vertex"
                elseif st.mode == TriangleMode
                    "Triangle"
                elseif st.mode == ActuatorMode
                    "Actuator"
                end
            end
        )

        deactivate_interaction!(ax, :rectanglezoom)

        mesh!(fig[1,1], 
            lift(state) do st
                if size(st.vertices)[2] < 3  || length(st.indices)<3
                    ones(2, 3) * NaN
                else
                    st.vertices
                end
            end, 
            lift(state) do st
                if length(st.indices)<3
                    [1, 2, 3]
                else
                    st.indices[1:(length(st.indices)÷3)*3]
                end
            end
        )

        scatter!(
            fig[1,1], 
            lift(state) do st
                st.vertices
            end
        )

        mesh!(
            fig[1,1], 
            lift(state) do st
                if length(st.actuators) < 1
                     ones(2, 3) * NaN
                else
                    vertices = Matrix{S}(undef, 2, 0)

                    for (i,j) in st.actuators
                        vi = st.vertices[:, i]
                        vj = st.vertices[:, j]

                        dir = normalize(vj - vi)

                        delta = [-dir[2],dir[1]] * 0.01

                        vertices = [vertices vi-delta vi+delta vj-delta vj+delta]
                    end

                    vertices
                end

            end,
            lift(state) do st
                if length(st.actuators) < 1
                     ones(Int64, 3)
                else
                    indices = Vector{I}()

                    for i in 1:length(st.actuators)
                        append!(indices, [0,-1,-2, 0, -2, -3] .+ 4*i)
                    end

                    indices
                end
            end
        )

        on(events(fig).keyboardbutton) do event
            st = state.val
            
            if event.key==Keyboard.v 
                switch_mode(st, VertexMode)
            elseif event.key==Keyboard.a
                switch_mode(st, ActuatorMode)
            elseif event.key==Keyboard.t
                switch_mode(st, TriangleMode)
            elseif event.key==Keyboard.escape
                switch_mode(st, ModifyMode)
            end
        end

        on(events(fig).mousebutton) do event
            on_click(state.val, event, Vector(mouseposition(ax.scene)))
        end

        state, fig
    end

    function on_mode_start(st::BuilderState)
        if st.mode == TriangleMode
            st.mode_state = Vector{I}()
        elseif st.mode == ActuatorMode
            st.mode_state = Vector{I}()
        end
    end

    function on_click(st::BuilderState, event, mpos)
        if st.mode == ModifyMode
                
        elseif st.mode == VertexMode
            if event.action == Mouse.press && event.button == Mouse.left
                add_vertex!(st, mpos)
            end
        elseif st.mode == TriangleMode
            if event.action == Mouse.press && event.button == Mouse.left
                vertex_found = false

                tri = convert(Vector{I}, st.mode_state)

                for i in 1:size(st.vertices)[2]
                    if norm(st.vertices[:,i] - mpos) < 0.05
                        push!(tri, i)
                        vertex_found = true
                        break
                    end
                end

                if !vertex_found
                    add_vertex!(st, mpos)
                    push!(tri, size(st.vertices)[2])
                end
                
                if length(tri) == 3
                    add_indices!(st, tri)
                    switch_mode(st, TriangleMode)
                end
            end
        elseif st.mode == ActuatorMode
            if event.action == Mouse.press && event.button == Mouse.left
                act = convert(Vector{I}, st.mode_state)

                for i in 1:size(st.vertices)[2]
                    if norm(st.vertices[:,i] - mpos) < 0.05
                        push!(act, i)
                        break
                    end
                end
                
                if length(act) == 2
                    
                    push!(st.actuators, (act[1], act[2]))
                    switch_mode(st, ActuatorMode)
                end
            end
        end
    end

    function switch_mode(st::BuilderState, mode::PlacementMode)
        on_mode_end(st)
        st.mode=mode
        on_mode_start(st)
    end

    function on_mode_end(st::BuilderState)
        display(st.actuators)
    end

    function redraw(state::Observable{BuilderState})
        notify(state)
    end

    function add_vertex!(state::BuilderState, vertex::Vector)
        state.vertices = [state.vertices vertex]
    end

    function add_indices!(state::BuilderState, indices::Vector{I})
        for index in indices
            push!(state.indices, index)
        end
    end
end

function test()
    builder, fig = SimulationBuilder.create()
    display(fig)


    while true
        sleep(0.1)
        SimulationBuilder.redraw(builder)
    end

end

test()