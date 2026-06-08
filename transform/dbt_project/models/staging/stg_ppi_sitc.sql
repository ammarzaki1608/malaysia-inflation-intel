with source as (

    select * from {{ source('raw', 'ppi_sitc')}}

),

cleaned as (

    select
        cast(date as date)          as date,
        cast(series as string)      as series,
        cast(section as string)     as section,
        cast(index as float64)      as ppi_index

    from source
    where date is not null 
        and series = 'abs'
        and section is not null
)

select * from cleaned
