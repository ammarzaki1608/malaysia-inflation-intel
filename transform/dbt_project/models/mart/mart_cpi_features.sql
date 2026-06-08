with cpi as (

    select
        date,
        state,
        division,
        cpi_index

    from {{ ref('stg_cpi_state') }}

),

ppi as (

    select
        date,
        ppi_index

    from {{ ref('stg_ppi_headline') }}

),

opr as (

    select
        date,
        opr_rate

    from {{ ref('stg_opr') }}

),

cpi_with_lags as (

    select
        date,
        state,
        division,
        cpi_index,

        lag(cpi_index, 1)  over (
            partition by state, division
            order by date
        ) as cpi_lag_1,

        lag(cpi_index, 3)  over (
            partition by state, division
            order by date
        ) as cpi_lag_3,

        lag(cpi_index, 6)  over (
            partition by state, division
            order by date
        ) as cpi_lag_6,

        lag(cpi_index, 12) over (
            partition by state, division
            order by date
        ) as cpi_lag_12,

        avg(cpi_index) over (
            partition by state, division
            order by date
            rows between 2 preceding and current row
        ) as cpi_ma_3,

        avg(cpi_index) over (
            partition by state, division
            order by date
            rows between 11 preceding and current row
        ) as cpi_ma_12,

        case
            when date between '2020-03-01' and '2021-12-31'
            then true
            else false
        end as is_covid_period

    from cpi

),

ppi_with_lags as (

    select
        date,
        ppi_index,

        lag(ppi_index, 1) over (
            order by date
        ) as ppi_lag_1,

        lag(ppi_index, 2) over (
            order by date
        ) as ppi_lag_2,

        lag(ppi_index, 3) over (
            order by date
        ) as ppi_lag_3

    from ppi

),

joined as (

    select
        c.date,
        c.state,
        c.division,
        c.cpi_index,
        c.cpi_lag_1,
        c.cpi_lag_3,
        c.cpi_lag_6,
        c.cpi_lag_12,
        c.cpi_ma_3,
        c.cpi_ma_12,
        c.is_covid_period,
        p.ppi_index,
        p.ppi_lag_1,
        p.ppi_lag_2,
        p.ppi_lag_3,
        o.opr_rate

    from cpi_with_lags c
    left join ppi_with_lags p on c.date = p.date
    left join opr           o on c.date = o.date

)

select * from joined