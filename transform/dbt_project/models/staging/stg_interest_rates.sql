with source as (
    select * from {{source ('raw', 'interest_rates')}}

),

cleaned as(

    select
        cast(date as date)          as date,
        cast(bank as string)        as bank,
        cast(rate as string)        as rate_type,
        cast(value as float64)      as rate_value

    from source
    where date is not null
        and value is not null 
)

select * from cleaned