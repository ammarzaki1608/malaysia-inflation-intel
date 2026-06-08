with source as (

    select * from {{ source('raw', 'hies_state')}}

),

cleaned as (

    select
        cast(date as date)                 as date,
        cast(state as string)              as state,
        cast(income_mean as float64)       as income_mean,
        cast(income_median as float64)     as income_median,
        cast(expenditure_mean as float64)  as expenditure_mean,
        cast(gini as float64)              as gini,
        cast(poverty as float64)           as poverty_rate,

        case
            when income_median < {{ var('b40_threshold') }} then 'B40'
            when income_median < {{ var('m40_threshold') }} then 'M40'
            else                                                 'T20'
        end as income_band

    from source
    where state is not null   
)

select * from cleaned