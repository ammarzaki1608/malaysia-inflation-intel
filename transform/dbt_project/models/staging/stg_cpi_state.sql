with source as (
    
    select * from {{ source('raw', 'cpi_state')}}

),

cleaned as (

    select
        cast(date as date)          as date,
        cast(state as string)       as state,
        cast(division as string)    as division,
        cast(index as FLOAT64)      as cpi_index

    from source
    where date is not null 
        and state is not null
        and division is not null
        and index is not null

)

select * from cleaned