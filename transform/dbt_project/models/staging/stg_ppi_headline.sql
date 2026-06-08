with source as (

    select * from {{ source('raw', 'ppi_headline')}}

),

cleaned as (

    select
        cast(date as date)          as date,
        cast(series as string)      as series,
        cast(index as float64)      as ppi_index,
        cast(index_sa as float64)   as ppi_index_sa

    from source
    where date is not null 
        and series = 'abs'

)

select * from cleaned